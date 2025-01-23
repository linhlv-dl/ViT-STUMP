import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pylab

import numpy as np

import os

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from step_function import StepFunction

def plot_survival_curves(rec_t, rec_e, antirec_t, antirec_e, experiment_name = '', output_file = None):
    # Set-up plots
    plt.figure(figsize=(12,3))
    ax = plt.subplot(111)

    # Fit survival curves
    kmf = KaplanMeierFitter()
    kmf.fit(rec_t, event_observed=rec_e, label=' '.join([experiment_name, "Recommendation"]))   
    kmf.plot(ax=ax,linestyle="-")
    kmf.fit(antirec_t, event_observed=antirec_e, label=' '.join([experiment_name, "Anti-Recommendation"]))
    kmf.plot(ax=ax,linestyle="--")
    
    # Format graph
    plt.ylim(0,1);
    ax.set_xlabel('Timeline (months)',fontsize='large')
    ax.set_ylabel('Percentage of Population Alive',fontsize='large')
    
    # Calculate p-value
    results = logrank_test(rec_t, antirec_t, rec_e, antirec_e, alpha=.95)
    results.print_summary()

    # Location the label at the 1st out of 9 tick marks
    xloc = max(np.max(rec_t),np.max(antirec_t)) / 9
    if results.p_value < 1e-5:
        ax.text(xloc,.2,'$p < 1\mathrm{e}{-5}$',fontsize=20)
    else:
        ax.text(xloc,.2,'$p=%f$' % results.p_value,fontsize=20)
    plt.legend(loc='best',prop={'size':15})


    if output_file:
        plt.tight_layout()
        pylab.savefig(output_file)

def _compute_counts(event, time):
    '''
    Parameters:
        - event: array = boolean event indicator
        - time: array = survival time or time of censoring
    Returns:
        - times: array = unique time points
        - n_events: array = number of events at each time point
        - n_at_risk: array = number of samples that have not been censored or have not had an event at each time point
        - n_censored: array = number of censored samples at each time point
    '''
    n_samples = event.shape[0]
    order = np.argsort(time, kind = 'mergesort')

    uniq_times = np.empty(n_samples, dtype = time.dtype)
    uniq_events = np.empty(n_samples, dtype = int)
    uniq_counts = np.empty(n_samples, dtype = int)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1
            count += 1
            i += 1
        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break
        prev_val = time[order[i]]
    times = np.resize(uniq_times, j)
    n_events = np.resize(uniq_events, j)
    total_count = np.resize(uniq_counts, j)
    n_censored = total_count - n_events

    # offset cumulative sum by one
    total_count = np.r_[0, total_count]
    n_at_risk = n_samples - np.cumsum(total_count)

    return times, n_events, n_at_risk[:-1], n_censored

def cox_log_rank(risk_preds, status, surv_times):
    '''
        Compute the Log_rank_cox (p_value)
        Parameters:
            - risk_preds: The risk predictions of all patients given by a trained survival model
            - status: the events of all patients (ground truth)
            - surv_times: The survival times of all patients (ground truth)
    '''
    risk_preds = risk_preds.cpu().numpy().reshape(-1)
    median = np.median(risk_preds)
    hazards_dichotomize = np.zeros([len(risk_preds)], dtype = int)
    hazards_dichotomize[risk_preds > median] = 1

    surv_times = surv_times.cpu().numpy().reshape(-1)
    idx = hazards_dichotomize == 0
    status = status.cpu().numpy()

    T1 = surv_times[idx]
    T2 = surv_times[~idx]
    E1 = status[idx]
    E2 = status[~idx]

    results = logrank_test(T1, T2, event_observed_A = E1, event_observed_B = E2)
    pvalue_pred = results.p_value

    return pvalue_pred
