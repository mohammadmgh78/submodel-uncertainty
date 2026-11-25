"""
Discrete-event simulator of the contact center (Section 6.3).

Two contact groups, three expert groups with two experts each: group 1 serves contact group 1, group 2 serves contact
group 2, and group 3 serves both. Contacts arrive, wait, abandon when their patience runs out, and are served. Routing
is triggered
  * when a contact can be assigned to a free expert (contact-triggered: choose the expert group), and
  * when an expert becomes available while contacts wait (expert-triggered: choose the contact group).

`Simulation` uses routing submodels (logistic regression or Bayesian neural network instances). An infeasible
prediction is replaced by a uniformly random feasible choice. `TrueSimulation` uses the true rules of the system.

Times are in minutes. Contacts whose departure falls after the end of the simulated horizon are not recorded.
"""
import copy
import heapq
import json
import os
import random

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HOURS = 9                    # length of the day
EPOCH = 30                   # minutes per epoch
CONTACT_GROUP = 2            # group of the KPI


# ---------------------------------------------------------------------------
# Routing-model prediction
# ---------------------------------------------------------------------------
class FastPredictor:
    """Predicted class label of a routing model for one predictor vector, memoised. Uses the same numerical operations
    as `model.predict` for LogisticRegression and TorchClassifierWrapper, without the per-call overhead."""

    def __init__(self, model):
        self.model = model
        self.cache = {}
        self._mode = 'generic'
        cls_name = type(model).__name__
        if cls_name == 'LogisticRegression' and hasattr(model, 'coef_'):
            self._mode = 'linear'
            self._coefT = model.coef_.T
            self._intercept = model.intercept_
            self._classes = model.classes_
        elif cls_name == 'TorchClassifierWrapper':
            self._mode = 'torch'
            import torch
            self._torch = torch
            self._net = model._rebuild()

    def _predict_generic(self, X):
        r = self.model.predict(X)
        return int(r[0]) if isinstance(r, np.ndarray) else int(r)

    def _predict_linear(self, X):
        Xf = np.ascontiguousarray(X, dtype=np.float64)
        scores = Xf @ self._coefT + self._intercept
        if scores.ndim > 1 and scores.shape[1] == 1:
            scores = np.reshape(scores, (-1,))
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.intp)
        else:
            indices = np.argmax(scores, axis=1)
        return int(np.take(self._classes, indices, axis=0)[0])

    def _predict_torch(self, X):
        torch = self._torch
        X_arr = np.array(X, dtype=np.float32)
        X_tensor = torch.tensor(X_arr)
        with torch.no_grad():
            out = self._net(X_tensor)
        if self.model.is_multiclass:
            result = (torch.argmax(out, dim=1) + 1).numpy()
        else:
            result = ((out > 0.5).float() + 1).numpy().flatten().astype(int)
        return int(result[0])

    def __call__(self, predictor_list):
        key = tuple(predictor_list)
        try:
            return self.cache[key]
        except (KeyError, TypeError):
            pass
        X = np.array(predictor_list).reshape(1, -1)
        if self._mode == 'linear':
            val = self._predict_linear(X)
        elif self._mode == 'torch':
            val = self._predict_torch(X)
        else:
            val = self._predict_generic(X)
        try:
            self.cache[key] = val
        except TypeError:
            pass
        return val


# ---------------------------------------------------------------------------
# Simulation objects
# ---------------------------------------------------------------------------
def schedule_builder(next_interval, this_interval, next_time, operation_hours):
    result = []
    for n, t in zip(next_interval, this_interval):
        if n == 0 and t == 0:
            result.append(0)
        elif n == 0 and t == 1:
            result.append(1)
        elif n == 0 and t == 2:
            result.append(0)
        elif n == 0 and t == 3:
            result.append(1)
        elif n in [0, 5, 6] and t in [5, 6]:
            result.append(n)
        elif n == 2 and t == 0:
            if next_time > operation_hours * 60:
                result.append(0)
            else:
                result.append(2)
        elif n == 2 and t == 1:
            if next_time > operation_hours * 60:
                result.append(1)
            else:
                result.append(3)
        elif n == 2 and t == 2:
            result.append(2)
        elif n == 2 and t == 3:
            if next_time > operation_hours * 60:
                result.append(1)
            else:
                result.append(3)
        elif n == 2 and t in [5, 6]:
            if next_time > operation_hours * 60:
                result.append(1)
            else:
                result.append(3)
        elif n == 2 and t == 4:
            result.append(4)
        elif n in [5, 6] and t == 0:
            result.append(n)
        elif n in [5, 6] and t == 1:
            if next_time > operation_hours * 60:
                result.append(1)
            else:
                result.append(n)
        else:
            result.append(None)
    return result


class Customer:
    customer_id = 1

    def __init__(self, arrival_time, service_time, patience_time, category):
        self.unique_id = Customer.customer_id
        self.category = category
        self.arrival = arrival_time
        self.service_time = service_time
        self.departure = 0
        self.time_in_queue_ = 0
        self.abandonment = 0
        self.patience_time = patience_time
        self.abandonment_time = self.arrival + patience_time
        self.served_expert_group_id = 0
        self.served_server_id = 0
        self.contact_thresholds = []
        Customer.customer_id += 1

    def time_in_queue(self, current_time):
        return current_time - self.arrival

    def __lt__(self, other):
        return self.arrival < other.arrival

    @classmethod
    def reset_customer_id(cls, cust_num_hot=1):
        cls.customer_id = cust_num_hot


class Experts:

    def __init__(self, server_details, hours_termination, minutes_shift):
        self.server_details = copy.deepcopy(server_details)
        for key in self.server_details.keys():
            d = self.server_details[key]
            d['Number_of_Servers'] = len(d['Schedules'])
            d['Call_ID_Served'] = set(d['Call_ID_Served'])
            d['status'] = [exp_det[1][0] for exp_det in d['Schedules']]
            d['time_last_changed'] = [0] * d['Number_of_Servers']
            d['server_utilization'] = [[] for n in range(d['Number_of_Servers'])]
            d['interval_utilization'] = [[] for n in range(d['Number_of_Servers'])]
            d['temp_utilization_storage'] = [[] for n in range(d['Number_of_Servers'])]
            d['interval_status'] = [exp_det[1] for exp_det in d['Schedules']]
            d['thresholds'] = []
            self.completed_tasks = 0

    def get_free_server(self, group_id, current_time):
        status = self.server_details[group_id]['status']
        idle_server_id = [i for i in range(self.server_details[group_id]['Number_of_Servers']) if status[i] == 0]
        random_index = int(random.random() * len(idle_server_id))
        return idle_server_id[random_index]

    def make_busy(self, current_time, group_id, server_id):
        server_id = int(server_id)
        self.server_details[group_id]['status'][server_id] = 1
        self.server_details[group_id]['time_last_changed'][server_id] = current_time

    def make_idle(self, current_time, group_id, server_id):
        if self.server_details[group_id]['status'][server_id] not in [3, 5, 6]:
            self.server_details[group_id]['status'][server_id] = 0
        if self.server_details[group_id]['status'][server_id] == 3:
            self.server_details[group_id]['status'][server_id] = 4
        self.server_details[group_id]['time_last_changed'][server_id] = current_time

    def keep_same(self, current_time, group_id, server_id):
        self.server_details[group_id]['time_last_changed'][server_id] = current_time

    def is_free(self, caller_type):
        for d in self.server_details.values():
            if caller_type in d['Call_ID_Served'] and 0 in d['status']:
                return True
        return False

    def queue_present(self, expert_group_id_temp, queue_detail_temp):
        for caller_type in self.server_details[expert_group_id_temp]['Call_ID_Served']:
            if queue_detail_temp[caller_type]['size'] > 0:
                return True
        return False

    def clear_storage_counter_server(self, current_time, operation_hours, minutes_change_shift):
        interval_num_tmp = int(current_time // minutes_change_shift)
        next_time = minutes_change_shift + current_time
        for key in self.server_details.keys():
            d = self.server_details[key]
            d['temp_utilization_storage'] = [[] for n in range(d['Number_of_Servers'])]
            this_interval = d['status']
            next_interval = [exp_det[1][interval_num_tmp] for exp_det in d['Schedules']]
            d['status'] = schedule_builder(next_interval, this_interval, next_time, operation_hours)

    def get_server_utilization(self, current_time, group_id, server_id):
        s = self.server_details[group_id]['status'][server_id]
        if s == 1:
            return current_time - self.server_details[group_id]['time_last_changed'][server_id]
        elif s == 3:
            return current_time - self.server_details[group_id]['time_last_changed'][server_id]
        else:
            return 0

    def any_server_active(self):
        return sum(sum(k['status']) for k in self.server_details.values()) > 0


class Queues:

    def __init__(self, num_caller_types):
        self.queue = {}
        for i in range(num_caller_types):
            self.queue[i + 1] = {'size': 0, 'time_last_changed': 0, 'NIQ': [], 'delay': [],
                                 'interval_NIQ': [], 'temp_NIQ': []}

    def add_to_queue(self, current_time, caller_type):
        self.queue[caller_type]['size'] += 1
        self.queue[caller_type]['time_last_changed'] = current_time

    def remove_from_queue(self, current_time, caller_type):
        self.queue[caller_type]['size'] -= 1
        self.queue[caller_type]['time_last_changed'] = current_time

    def keep_same(self, current_time, caller_type):
        self.queue[caller_type]['time_last_changed'] = current_time

    def number_in_queue(self, current_time, caller_type):
        return self.queue[caller_type]['size'] * (current_time - self.queue[caller_type]['time_last_changed'])

    def clear_storage_counter(self):
        for i in self.queue.keys():
            self.queue[i]['temp_NIQ'] = []

    def are_queues_present(self):
        return sum([queue['size'] for queue in self.queue.values()]) > 0


class Event:
    """Event calendar (a heap). `_thr` records the times of each customer's Threshold events in the calendar, so that
    they can be located quickly when they are cancelled."""

    def __init__(self):
        self.calender = []
        self._thr = {}

    def schedule_arrival(self, arrival_time, obj):
        heapq.heappush(self.calender, (arrival_time, 'Arrival', obj))

    def schedule_departure(self, departure_time, obj):
        heapq.heappush(self.calender, (departure_time, 'Departure', obj))

    def schedule_abandonment(self, abandonment_time, obj):
        heapq.heappush(self.calender, (abandonment_time, 'Abandonment', obj))

    def schedule_interval_collection_time(self, intervals):
        for i in intervals:
            heapq.heappush(self.calender, (i, 'Interval End', ' '))

    def schedule_interval_change_shift(self, intervals):
        for i in intervals:
            heapq.heappush(self.calender, (i, 'Interval End Shift', ' '))

    def schedule_threshold_check(self, closest_threshold, obj):
        heapq.heappush(self.calender, (closest_threshold, 'Threshold', obj))
        self._thr.setdefault(obj, []).append(closest_threshold)

    def _forget_threshold(self, obj, t):
        lst = self._thr[obj]
        lst.remove(t)
        if not lst:
            del self._thr[obj]

    def next_event(self):
        time, event_type, obj = heapq.heappop(self.calender)
        if event_type == 'Threshold':
            self._forget_threshold(obj, time)
        return time, event_type, obj

    def remove_abandonment(self, obj):
        """Cancel the customer's abandonment event."""
        try:
            idx = self.calender.index((obj.abandonment_time, 'Abandonment', obj))
        except ValueError:
            return
        del self.calender[idx]
        heapq.heapify(self.calender)

    def remove_threshold(self, obj):
        """Cancel the first of the customer's threshold events."""
        times = self._thr.get(obj)
        if not times:
            return
        cal = self.calender
        idx = min(cal.index((t, 'Threshold', obj)) for t in times)
        t_removed = cal[idx][0]
        del cal[idx]
        heapq.heapify(cal)
        self._forget_threshold(obj, t_removed)


class Routing:
    def __init__(self):
        self.routing_dict = dict()
        self.routing_dict['routing contact'] = dict()
        self.routing_dict['routing contact']['contact'] = []
        self.routing_dict['routing contact']['expert'] = []
        self.routing_dict['routing contact']['experts free lengths'] = []
        self.routing_dict['routing expert'] = dict()
        self.routing_dict['routing expert']['contact'] = []
        self.routing_dict['routing expert']['expert'] = []
        self.routing_dict['routing expert']['contacts queue lengths'] = []
        self.routing_dict['routing expert']['contacts waiting times'] = []

    def create_datasets(self):
        datasets = {}
        for key in ['routing contact', 'routing expert']:
            data = self.routing_dict[key]
            df = pd.DataFrame.from_dict(data, orient='index').transpose()
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, list) and any(isinstance(i, list) for i in x)).any():
                    max_len = max(len(sublist) for sublist in df[col] if isinstance(sublist, list))
                    for i in range(max_len):
                        df[f'{col}_{i+1}'] = df[col].apply(lambda x: x[i] if i < len(x) else None)
                    df.drop(columns=[col], inplace=True)
            datasets[key] = df
        return datasets


class ThresholdTable:
    """Waiting-time thresholds after which an expert group can serve a contact group."""

    def __init__(self, threshold_df):
        contacts = threshold_df['Contact'].to_numpy()
        experts = threshold_df['Expert'].to_numpy()
        thr = threshold_df['Threshold'].to_numpy()
        # per contact, in table order: (expert, threshold)
        self.by_contact = {}
        self.thr_vals = {}
        for c, e, t in zip(contacts, experts, thr):
            self.by_contact.setdefault(c, []).append((e, t))
            self.thr_vals.setdefault(c, []).append(t)
        # the scalar type (Python float or np.float64) returned by pandas matters for the later round()
        s = threshold_df['Threshold']
        cast_series_min = type(s.min())
        cast_builtin_min = type(min(s))
        self.thr_series_min = {c: [cast_series_min(t) for t in v] for c, v in self.thr_vals.items()}
        self.thr_builtin_min = {c: [cast_builtin_min(t) for t in v] for c, v in self.thr_vals.items()}
        # per expert group: (contact group, threshold) sorted by contact group
        self.by_expert = {}
        for e in pd.unique(experts):
            sub = threshold_df[threshold_df['Expert'] == e].sort_values(by='Contact', ascending=True)
            self.by_expert[int(e)] = list(zip(sub['Contact'].to_numpy(), sub['Threshold'].to_numpy()))

    def experts_reached(self, contact, waiting_time):
        """Expert groups whose threshold for this contact group has been reached."""
        return [e for e, t in self.by_contact.get(contact, ()) if t <= waiting_time]

    def next_threshold(self, contact, waiting_time):
        """Next threshold after the current waiting time, or None."""
        cands = [t for t in self.thr_series_min.get(contact, ()) if t > waiting_time]
        if not cands:
            return None
        return min(cands)

    def first_threshold(self, contact):
        """Smallest threshold of this contact group."""
        return min(self.thr_builtin_min[contact])


class Simulation:

    def __init__(self, hours_operating, hours_termination, minutes_data_collection,
                 minutes_shift, contact_datasets, server_details, threshold_table, initial_state, hot_start,
                 predictor_contact=None, predictor_expert=None):
        self.sim_clock = hot_start
        self.minutes_data_collection = minutes_data_collection
        self.minutes_shift = minutes_shift
        self.day_length = hours_operating * 60
        self.sim_hours_termination = hours_termination
        self.intervals_data_collection = np.arange(0, hours_termination * 60 + 1, minutes_data_collection)[1:]
        self.intervals_shift = np.arange(0, hours_termination * 60 + 1, minutes_shift)[1:]
        self.hours_operating = hours_operating
        self.no_of_categories = len(list(contact_datasets['Contact Group'].unique()))

        self.next_event = 0
        self.next_event_time = 0
        self.call_obj = 0

        self.routing_obj = Routing()
        self.contact_datasets = contact_datasets
        self.thr = threshold_table
        self.arrival_times = [[] for _ in range(self.no_of_categories)]
        self.hot_start = hot_start
        self.state_all = initial_state
        self.predict_contact = predictor_contact
        self.predict_expert = predictor_expert

        self.callers_list = []
        self.arrival_counts = [[0, 0] for _ in range(int(self.hours_operating * 60 // self.minutes_shift))]
        self.queue_ = Queues(self.no_of_categories)
        self.server = Experts(server_details, hours_termination, self.minutes_shift)
        self.event_ = Event()

        # positional pairing in check_routing/routing requires this
        for g, d in self.server.server_details.items():
            if len(self.thr.by_expert.get(int(g), [])) != len(d['Call_ID_Served']):
                raise ValueError(f'threshold rows for expert group {g} do not match its Call_ID_Served')

    def store_callers(self, obj):
        self.callers_list.append(obj)

    def generate_arrivals(self):
        time = self.hot_start
        if self.hot_start != 0:
            for idx, row in self.state_all[0].iterrows():
                self.call_obj = Customer(row['Arrival Time'], row['Handle Time'], row['Patience Time'],
                                         int(row['Contact Group']))
                self.call_obj.unique_id = row['Contact ID']
                self.event_.schedule_abandonment(self.call_obj.abandonment_time, self.call_obj)
                heapq.heappush(self.arrival_times[int(row['Contact Group']) - 1], [self.call_obj.arrival, self.call_obj])
                self.queue_.add_to_queue(self.hot_start, self.call_obj.category)
            for idx, row in self.state_all[1].iterrows():
                self.call_obj = Customer(row['Arrival Time'], row['Handle Time'], row['Patience Time'],
                                         int(row['Contact Group']))
                self.call_obj.unique_id = row['Contact ID']
                if row['Event Type'] == 'Departure':
                    self.call_obj.served_expert_group_id = str(int(float(row['Expert Group'])))
                    self.call_obj.served_server_id = int(row['Expert ID'])
                    self.event_.schedule_departure(self.hot_start - row['Time Elapsed'] + row['Handle Time'], self.call_obj)
                    self.server.make_busy(self.hot_start, self.call_obj.served_expert_group_id, self.call_obj.served_server_id)
            cust_num_hot = max(max(self.state_all[0]['Contact ID']), max(self.state_all[1]['Contact ID']))
        else:
            cust_num_hot = 1
        contact_dataset = self.contact_datasets
        contact_dataset = contact_dataset.sort_values(by='Arrival Time')
        contact_dataset = contact_dataset[contact_dataset['Arrival Time'] > self.hot_start]
        Customer.reset_customer_id(cust_num_hot=cust_num_hot + 1)

        cols = list(contact_dataset.columns)
        rows = contact_dataset.to_numpy()
        i_arr, i_ht, i_pt, i_cg = (cols.index(c) for c in ('Arrival Time', 'Handle Time', 'Patience Time', 'Contact Group'))
        term = self.sim_hours_termination * 60
        for r in rows:
            self.call_obj = Customer(r[i_arr], r[i_ht], r[i_pt], int(r[i_cg]))
            time = self.call_obj.arrival
            self.event_.schedule_arrival(self.call_obj.arrival, self.call_obj)
            self.event_.schedule_abandonment(self.call_obj.abandonment_time, self.call_obj)
            if time > term:
                break
            self.arrival_counts[int(r[i_arr] // self.minutes_shift)][int(r[i_cg] - 1)] += 1

    # ---- helpers ----------------------------------------------------------
    def _free_counts_reached(self, caller_type, reached):
        """Number of free experts per group; 0 for groups that cannot serve this contact (yet)."""
        out = []
        for expert_id, d in self.server.server_details.items():
            if caller_type in d['Call_ID_Served'] and int(expert_id) in reached:
                out.append(np.int64(d['status'].count(0)))
            else:
                out.append(0)
        return out

    def _head_arrival_times(self):
        return [self.arrival_times[contact_id - 1][0][0] if self.arrival_times[contact_id - 1] else self.sim_clock
                for contact_id in self.queue_.queue.keys()]

    def _schedule_next_threshold_for_head(self, category):
        """If the queue of `category` is non-empty, schedule its head's next threshold check."""
        head_list = self.arrival_times[category - 1]
        if head_list != []:
            waiting_time_temp_next = round(self.sim_clock - head_list[0][0], 6)
            next_threshold = self.thr.next_threshold(category, waiting_time_temp_next)
            if next_threshold is not None:
                self.event_.schedule_threshold_check(round(next_threshold + head_list[0][0], 6), head_list[0][1])

    # ---- routing ----------------------------------------------------------
    def check_routing_when_free(self, caller_type):
        waiting_time_contact = round(self.sim_clock - self.call_obj.arrival, 6)
        reached = self.thr.experts_reached(caller_type, waiting_time_contact)
        if not reached:
            return False
        for expert_id, d in self.server.server_details.items():
            if caller_type in d['Call_ID_Served'] and int(expert_id) in reached and 0 in d['status']:
                return True
        return False

    def routing_when_free(self, caller_type):
        waiting_time_contact = round(self.sim_clock - self.call_obj.arrival, 6)
        reached = self.thr.experts_reached(caller_type, waiting_time_contact)
        num_of_free_experts = self._free_counts_reached(caller_type, reached)

        chosen_expert = self.predict_contact([int(caller_type), *num_of_free_experts])
        if num_of_free_experts[chosen_expert - 1] == 0 or caller_type not in self.server.server_details[str(chosen_expert)]['Call_ID_Served']:
            # infeasible prediction: choose uniformly among the feasible expert groups
            chosen_expert = 1 + random.choice([i for i, val in enumerate(num_of_free_experts)
                                               if val != 0 and caller_type in self.server.server_details[str(i + 1)]['Call_ID_Served']])
        self.routing_obj.routing_dict['routing contact']['contact'].append(caller_type)
        self.routing_obj.routing_dict['routing contact']['expert'].append(chosen_expert)
        self.routing_obj.routing_dict['routing contact']['experts free lengths'].append(num_of_free_experts)
        return str(chosen_expert)

    def check_routing(self, group_id):
        if not self.server.queue_present(group_id, self.queue_.queue):
            return False
        arrival_times_list = self._head_arrival_times()
        served = self.server.server_details[group_id]['Call_ID_Served']
        waiting_times_list = [round(self.sim_clock - arrival_times_list[j], 6) for j in range(len(arrival_times_list)) if j + 1 in served]
        for (c, t), w in zip(self.thr.by_expert[int(group_id)], waiting_times_list):
            if t < w:
                return True
        return False

    def routing(self, group_id):
        self.sim_clock = self.next_event_time
        queues = self.queue_.queue
        queues_to_check = self.server.server_details[group_id]['Call_ID_Served']
        queue_lengths = [queues[contact_id]['size'] for contact_id in queues.keys()]
        arrival_times_list = self._head_arrival_times()
        waiting_times_list = [round(self.sim_clock - arrival_times_list[j], 6) for j in range(len(arrival_times_list)) if j + 1 in queues_to_check]

        eligible_contacts = [c for (c, t), w in zip(self.thr.by_expert[int(group_id)], waiting_times_list) if t < w]
        waiting_times_all = [round(self.sim_clock - a, 6) for a in arrival_times_list]
        chosen_contact = self.predict_expert([int(group_id), *queue_lengths, *waiting_times_all])
        if chosen_contact not in self.server.server_details[str(group_id)]['Call_ID_Served'] or queue_lengths[chosen_contact - 1] == 0:
            # infeasible prediction: choose uniformly among the feasible contact groups
            chosen_contact = random.choice([i for i in eligible_contacts if queues[i]['size'] > 0])

        self.routing_obj.routing_dict['routing expert']['expert'].append(group_id)
        self.routing_obj.routing_dict['routing expert']['contacts queue lengths'].append(queue_lengths)
        self.routing_obj.routing_dict['routing expert']['contacts waiting times'].append(waiting_times_list)
        self.routing_obj.routing_dict['routing expert']['contact'].append(chosen_contact)
        return chosen_contact

    # ---- events -----------------------------------------------------------
    def change_shift(self):
        self.sim_clock = self.next_event_time
        contact_served_flag = [0 for _ in range(len(self.queue_.queue))]
        for expert_group_temp in self.server.server_details.keys():
            for expert_status_temp in range(len(self.server.server_details[expert_group_temp]['status'])):
                if self.server.server_details[expert_group_temp]['status'][expert_status_temp] == 0 and self.check_routing(expert_group_temp):
                    next_queue_no = self.routing(expert_group_temp)
                    self.call_obj = heapq.heappop(self.arrival_times[next_queue_no - 1])[1]
                    self.queue_.queue[self.call_obj.category]['delay'].append(self.call_obj.time_in_queue_)
                    self.queue_.remove_from_queue(self.next_event_time, self.call_obj.category)
                    self.event_.remove_abandonment(self.call_obj)
                    self.event_.remove_threshold(self.call_obj)
                    self.call_obj.served_expert_group_id = expert_group_temp
                    self.call_obj.served_server_id = expert_status_temp
                    self.sim_clock = self.next_event_time
                    self.call_obj.departure = self.call_obj.service_time + self.sim_clock
                    self.event_.schedule_departure(self.call_obj.departure, self.call_obj)
                    contact_served_flag[self.call_obj.category - 1] = 1
                    self.server.make_busy(self.next_event_time, self.call_obj.served_expert_group_id, self.call_obj.served_server_id)
        for contact_group_temp in range(len(self.queue_.queue)):
            if contact_served_flag[contact_group_temp] == 1:
                self._schedule_next_threshold_for_head(contact_group_temp + 1)

    def threshold_check(self):
        self.sim_clock = self.next_event_time
        waiting_time_temp = round(self.sim_clock - self.call_obj.arrival, 6)
        if self.check_routing_when_free(self.call_obj.category):
            self.call_obj.served_expert_group_id = self.routing_when_free(self.call_obj.category)
            self.call_obj.served_server_id = self.server.get_free_server(self.call_obj.served_expert_group_id, self.next_event_time)
            g, s = self.call_obj.served_expert_group_id, self.call_obj.served_server_id
            self.server.server_details[g]['server_utilization'][s].append(self.server.get_server_utilization(self.next_event_time, g, s))
            self.server.server_details[g]['temp_utilization_storage'][s].append(self.server.get_server_utilization(self.next_event_time, g, s))
            self.server.make_busy(self.next_event_time, g, s)
            self.queue_.queue[self.call_obj.category]['NIQ'].append(self.queue_.number_in_queue(self.next_event_time, self.call_obj.category))
            self.queue_.queue[self.call_obj.category]['temp_NIQ'].append(self.queue_.number_in_queue(self.next_event_time, self.call_obj.category))
            self.queue_.remove_from_queue(self.next_event_time, self.call_obj.category)
            self.sim_clock = self.next_event_time
            heapq.heappop(self.arrival_times[self.call_obj.category - 1])
            self.call_obj.departure = self.call_obj.service_time + self.sim_clock
            self.event_.schedule_departure(self.call_obj.departure, self.call_obj)
            self.event_.remove_abandonment(self.call_obj)
            self._schedule_next_threshold_for_head(self.call_obj.category)
        else:
            next_threshold = self.thr.next_threshold(self.call_obj.category, waiting_time_temp)
            if next_threshold is not None:
                self.event_.schedule_threshold_check(round(next_threshold + self.call_obj.arrival, 6), self.call_obj)

    def arrival(self):
        self.sim_clock = self.next_event_time
        heapq.heappush(self.arrival_times[self.call_obj.category - 1], [self.call_obj.arrival, self.call_obj])
        if self.server.is_free(self.call_obj.category) and self.check_routing_when_free(self.call_obj.category):
            self.call_obj.served_expert_group_id = self.routing_when_free(self.call_obj.category)
            self.call_obj.served_server_id = self.server.get_free_server(self.call_obj.served_expert_group_id, self.next_event_time)
            g, s = self.call_obj.served_expert_group_id, self.call_obj.served_server_id
            self.server.server_details[g]['server_utilization'][s].append(self.server.get_server_utilization(self.next_event_time, g, s))
            self.server.server_details[g]['temp_utilization_storage'][s].append(self.server.get_server_utilization(self.next_event_time, g, s))
            self.server.make_busy(self.next_event_time, g, s)
            self.queue_.queue[self.call_obj.category]['NIQ'].append(self.queue_.number_in_queue(self.next_event_time, self.call_obj.category))
            self.queue_.queue[self.call_obj.category]['temp_NIQ'].append(self.queue_.number_in_queue(self.next_event_time, self.call_obj.category))
            self.call_obj.time_in_queue_ = 0
            self.queue_.queue[self.call_obj.category]['delay'].append(self.call_obj.time_in_queue_)
            self.sim_clock = self.next_event_time
            heapq.heappop(self.arrival_times[self.call_obj.category - 1])
            self.call_obj.departure = self.call_obj.service_time + self.sim_clock
            self.event_.schedule_departure(self.call_obj.departure, self.call_obj)
            self.event_.remove_abandonment(self.call_obj)
        elif self.server.is_free(self.call_obj.category) and self.queue_.queue[self.call_obj.category]['size'] == 0:
            closest_threshold = self.thr.first_threshold(self.call_obj.category)
            self.event_.schedule_threshold_check(round(self.call_obj.arrival + closest_threshold, 6), self.call_obj)
            self.queue_.queue[self.call_obj.category]['NIQ'].append(self.queue_.number_in_queue(self.next_event_time, self.call_obj.category))
            self.queue_.queue[self.call_obj.category]['temp_NIQ'].append(self.queue_.number_in_queue(self.next_event_time, self.call_obj.category))
            self.queue_.add_to_queue(self.next_event_time, self.call_obj.category)
            self.sim_clock = self.next_event_time
        else:
            if self.queue_.queue[self.call_obj.category]['size'] == 0:
                closest_threshold = self.thr.first_threshold(self.call_obj.category)
                self.event_.schedule_threshold_check(round(self.call_obj.arrival + closest_threshold, 6), self.call_obj)
            self.queue_.queue[self.call_obj.category]['NIQ'].append(self.queue_.number_in_queue(self.next_event_time, self.call_obj.category))
            self.queue_.queue[self.call_obj.category]['temp_NIQ'].append(self.queue_.number_in_queue(self.next_event_time, self.call_obj.category))
            self.queue_.add_to_queue(self.next_event_time, self.call_obj.category)
            self.sim_clock = self.next_event_time

    def departure(self):
        self.call_obj.time_in_queue_ = round(self.next_event_time - self.call_obj.arrival - self.call_obj.service_time, 6)
        self.sim_clock = self.next_event_time
        group_id = self.call_obj.served_expert_group_id
        g, s = self.call_obj.served_expert_group_id, self.call_obj.served_server_id
        if (self.server.server_details[g]['status'][s] in [3, 4, 5, 6] or self.check_routing(group_id) == False):
            self.server.server_details[g]['server_utilization'][s].append(self.server.get_server_utilization(self.next_event_time, g, s))
            self.server.server_details[g]['temp_utilization_storage'][s].append(self.server.get_server_utilization(self.next_event_time, g, s))
            self.server.make_idle(self.next_event_time, g, s)
            self.queue_.queue[self.call_obj.category]['NIQ'].append(self.queue_.number_in_queue(self.next_event_time, self.call_obj.category))
            self.queue_.queue[self.call_obj.category]['temp_NIQ'].append(self.queue_.number_in_queue(self.next_event_time, self.call_obj.category))
            self.sim_clock = self.next_event_time
            self.store_callers(self.call_obj)
        else:
            self.store_callers(self.call_obj)
            group_id = self.call_obj.served_expert_group_id
            idle_server_id = self.call_obj.served_server_id
            self.queue_.queue[self.call_obj.category]['NIQ'].append(self.queue_.number_in_queue(self.next_event_time, self.call_obj.category))
            self.queue_.queue[self.call_obj.category]['temp_NIQ'].append(self.queue_.number_in_queue(self.next_event_time, self.call_obj.category))
            next_queue_no = self.routing(group_id)
            self.call_obj = heapq.heappop(self.arrival_times[next_queue_no - 1])[1]
            self.queue_.queue[self.call_obj.category]['delay'].append(self.call_obj.time_in_queue_)
            self.queue_.remove_from_queue(self.next_event_time, self.call_obj.category)
            self.event_.remove_abandonment(self.call_obj)
            self.event_.remove_threshold(self.call_obj)
            self.call_obj.served_expert_group_id = group_id
            self.call_obj.served_server_id = idle_server_id
            self.sim_clock = self.next_event_time
            self.call_obj.departure = self.call_obj.service_time + self.sim_clock
            self.event_.schedule_departure(self.call_obj.departure, self.call_obj)
            self._schedule_next_threshold_for_head(self.call_obj.category)

    def abandonment(self):
        self.call_obj.abandonment = 1
        self.call_obj.departure = self.call_obj.abandonment_time
        self.call_obj.time_in_queue_ = round(self.call_obj.abandonment_time - self.call_obj.arrival, 6)
        self.queue_.queue[self.call_obj.category]['NIQ'].append(self.queue_.number_in_queue(self.next_event_time, self.call_obj.category))
        self.queue_.queue[self.call_obj.category]['temp_NIQ'].append(self.queue_.number_in_queue(self.next_event_time, self.call_obj.category))
        self.queue_.remove_from_queue(self.next_event_time, self.call_obj.category)
        self.store_callers(self.call_obj)
        counter_temp = 0
        q = self.arrival_times[self.call_obj.category - 1]
        if len(q) > 0:
            for k in q:
                if k[1] == self.call_obj:
                    q.remove(k)
                    heapq.heapify(q)
                    break
                counter_temp += 1
        self.event_.remove_threshold(self.call_obj)
        self.sim_clock = self.next_event_time
        if counter_temp == 0:
            self._schedule_next_threshold_for_head(self.call_obj.category)

    def calculate_metrics(self):
        for key in self.queue_.queue.keys():
            self.queue_.queue[key]['NIQ'].append(self.queue_.number_in_queue(self.next_event_time, key))
            self.queue_.queue[key]['temp_NIQ'].append(self.queue_.number_in_queue(self.next_event_time, key))
            self.queue_.queue[key]['interval_NIQ'].append(sum(self.queue_.queue[key]['temp_NIQ']))
            self.queue_.keep_same(self.next_event_time, key)
        self.sim_clock = self.next_event_time
        self.queue_.clear_storage_counter()

    def change_shift_all(self):
        for key in self.server.server_details.keys():
            d = self.server.server_details[key]
            for i in range(len(d['temp_utilization_storage'])):
                d['server_utilization'][i].append(self.server.get_server_utilization(self.next_event_time, key, i))
                d['temp_utilization_storage'][i].append(self.server.get_server_utilization(self.next_event_time, key, i))
                d['interval_utilization'][i].append(sum(d['temp_utilization_storage'][i]))
                self.server.keep_same(self.next_event_time, key, i)
        self.sim_clock = self.next_event_time
        self.server.clear_storage_counter_server(self.sim_clock, self.hours_operating, self.minutes_shift)
        self.change_shift()

    def run(self):
        """Simulate until the end of the horizon; returns the customers who left the system."""
        self.event_.schedule_interval_collection_time(self.intervals_data_collection)
        self.event_.schedule_interval_change_shift(self.intervals_shift)
        self.generate_arrivals()
        handlers = {'Interval End': self.calculate_metrics, 'Interval End Shift': self.change_shift_all,
                    'Arrival': self.arrival, 'Departure': self.departure, 'Threshold': self.threshold_check}
        while self.event_.calender != []:
            self.next_event_time, self.next_event_type, self.call_obj = self.event_.next_event()
            if self.next_event_time > self.sim_hours_termination * 60:
                break
            handlers.get(self.next_event_type, self.abandonment)()
        return self.callers_list


class TrueSimulation(Simulation):
    """The real system: rule-based routing."""

    def routing_when_free(self, caller_type):
        """Contact-triggered: contact group 1 prefers expert group 1 to 3; contact group 2 prefers expert group 3 to 2."""
        details = self.server.server_details
        free = [int(g) for g, d in details.items() if caller_type in d['Call_ID_Served'] and 0 in d['status']]
        chosen = free[0] if len(free) == 1 else {1: 1, 2: 3}[caller_type]
        rec = self.routing_obj.routing_dict['routing contact']
        rec['contact'].append(caller_type)
        rec['expert'].append(chosen)
        rec['experts free lengths'].append([d['status'].count(0) for d in details.values()])
        return str(chosen)

    def routing(self, group_id):
        """Expert-triggered: groups 1 and 2 serve a single contact group; group 3 compares the two queues."""
        self.sim_clock = self.next_event_time
        queues = self.queue_.queue
        eligible = [c for c in self.server.server_details[group_id]['Call_ID_Served'] if queues[c]['size'] > 0]
        q1, q2 = (queues[c]['size'] for c in (1, 2))
        w1, w2 = waits = [self.sim_clock - a for a in self._head_arrival_times()]
        if len(eligible) == 1:
            chosen = eligible[0]
        elif q1 > 3 and w1 < 0.015:
            chosen = 1
        elif q2 < q1 and w2 < 10:
            chosen = 2
        elif q1 == q2:
            chosen = 1 if w1 < w2 else 2
        else:
            chosen = 1
        rec = self.routing_obj.routing_dict['routing expert']
        rec['expert'].append(group_id)
        rec['contacts queue lengths'].append([q1, q2])
        rec['contacts waiting times'].append(waits)
        rec['contact'].append(chosen)
        return chosen


def load_thresholds():
    t = pd.read_csv(os.path.join(HERE, 'inputs', 'threshold_dataset.csv'))
    return ThresholdTable(t.rename(columns={'Contact Group': 'Contact', 'Expert Group': 'Expert'}).sort_values(by='Expert'))


def load_experts():
    """Expert groups: schedules (one activity code per half hour; 0 = available, 2 = off) and contact groups served."""
    with open(os.path.join(HERE, 'inputs', 'expert_schedules.json')) as f:
        schedules = json.load(f)
    served = {'1': [1], '2': [2], '3': [1, 2]}
    return {g: {'Schedules': schedules[g], 'Call_ID_Served': served[g]} for g in schedules}


def load_model(name):
    """Predictor of a saved routing submodel (models/<name>.pkl)."""
    return FastPredictor(joblib.load(os.path.join(HERE, 'models', f'{name}.pkl')))


def run_replications(contacts, initial_states, hot_start, models=None, hours=HOURS):
    """Simulate one replication per entry of `contacts` ({rep: DataFrame}) from `initial_states` ({rep: [queue,
    service]}) at time `hot_start`. models = (contact-side, expert-side) predictors, or None for the true rules.
    Returns the Simulation objects after the run, by replication."""
    thresholds, experts = load_thresholds(), load_experts()
    sims = {}
    for rep in contacts:
        args = (hours, EPOCH, EPOCH, contacts[rep], experts, thresholds, initial_states[rep], hot_start)
        sim = TrueSimulation(HOURS, *args) if models is None else Simulation(HOURS, *args, *models)
        sim.run()
        sims[rep] = sim
    return sims


def customers_frame(sim, rep):
    """Records of the customers who left the system."""
    def expert_id(c):
        g = int(c.served_expert_group_id)
        return 0 if g == 0 else sim.server.server_details[str(g)]['Schedules'][c.served_server_id][0]
    return pd.DataFrame([{'ID': c.unique_id, 'Rep': rep, 'Contact Group': c.category,
                          'Expert Group': int(c.served_expert_group_id), 'Expert ID': expert_id(c),
                          'Interval Arrived': c.arrival // EPOCH + 1, 'Arrival Time': c.arrival,
                          'Interval Service Begin': (c.arrival + c.time_in_queue_) // EPOCH + 1,
                          'Handle Time': c.service_time, 'Patience Time': c.patience_time, 'Departure': c.departure,
                          'Abandon': c.abandonment, 'Waiting Time': c.time_in_queue_} for c in sim.callers_list])


def kpi(sim, epoch):
    """Mean waiting time of the contacts of group 2 that arrive in the given epoch (1, 2, ...); None if there are none."""
    w = [c.time_in_queue_ for c in sim.callers_list if c.category == CONTACT_GROUP and c.arrival // EPOCH + 1 == epoch]
    return float(pd.Series(w).groupby([0] * len(w)).mean().iloc[0]) if w else None
