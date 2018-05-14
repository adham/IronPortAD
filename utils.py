import os
import scipy
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

def find_files(pin, tag):
    """finds file names in `pin` that have `tag`
    """

    file_names = os.listdir(pin)
    file_names = [ff for ff in file_names if tag in ff]
    file_names = [os.path.join(pin, ff) for ff in file_names]
    file_names.sort()

    return file_names


def get_date_data(data, date):
    date_data = [dd for dd in data if dd[0].date() == date.date()]
    return date_data

def get_int_ext_porportion(data):
    emails = [email for dd in data for email in dd[1]]
    is_internal = ['telstra' in email for email in emails]
    internal = sum(is_internal) / len(is_internal)
    external = 1-internal
    return internal, external


def get_features(data):
    """extracts features from data of a date
    """

    features = []

    # 1. sent features
    sent_data = data['sent']

    # number of sent emails
    features.append(len(sent_data))

    # min, max, mean, std size of emails
    sizes = np.array([dd[2] for dd in sent_data])
    if len(sizes) != 0:
        features.append(sizes.min())
        features.append(sizes.max())
        features.append(sizes.mean())
        features.append(sizes.std())
    else:
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)

    # min, max, mean, std number of recipients
    n_recipients = np.array([len(dd[1]) for dd in sent_data])
    if len(n_recipients) != 0:
        features.append(n_recipients.min())
        features.append(n_recipients.max())
        features.append(n_recipients.mean())
        features.append(n_recipients.std())

        # number of unique recipitns
        features.append(np.unique(n_recipients).shape[0])

    else:
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)


    # ratio of internal emails to external emails
    if len(sent_data) != 0:
        int_porportion, ext_porportion = get_int_ext_porportion(sent_data)
        features.append(int_porportion)
        features.append(ext_porportion)

        # portion of emails inside/outside working hours
        sent_in_working_hrs = sum([dd[0].hour > 8 and dd[0].hour < 18 for dd in sent_data])
        tmp = sent_in_working_hrs/len(sent_data)
        features.append(tmp)
        features.append(1-tmp)

    else:
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)

    # len received,
    # 2. received features
    recieved_data = data['received']

    # number of received emails
    features.append(len(recieved_data))

    # min, max, mean, std size of emails
    if len(recieved_data) != 0:
        sizes = np.array([dd[2] for dd in recieved_data])
        features.append(sizes.min())
        features.append(sizes.max())
        features.append(sizes.mean())
        features.append(sizes.std())

        # number of unique senders
        senders = [dd[1] for dd in recieved_data]
        features.append(len(set(senders)))

        from_telstra = ['telstra' in ss for ss in senders]
        tmp = sum(from_telstra)/len(senders)
        features.append(tmp)
        features.append(1-tmp)

    else:
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)


    return features

def get_features(data):
    """extracts features from data of a date
    """
    # features
    # len(sent_data), min/max/mean/std email size,  min/max/mean/std number of recipients
    # ratio of internal and external emails, proportion of email sent in working hours
    # outside working hours

    features = []

    # 1. sent features
    sent_data = data['sent']

    # number of sent emails
    features.append(len(sent_data))

    # min, max, mean, std size of emails
    sizes = np.array([dd[2] for dd in sent_data])
    if len(sizes) != 0:
        features.append(sizes.min())
        features.append(sizes.max())
        features.append(sizes.mean())
        features.append(sizes.std())
    else:
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)

    # min, max, mean, std number of recipients
    n_recipients = np.array([len(dd[1]) for dd in sent_data])
    if len(n_recipients) != 0:
        features.append(n_recipients.min())
        features.append(n_recipients.max())
        features.append(n_recipients.mean())
        features.append(n_recipients.std())

        # number of unique recipitns
        features.append(np.unique(n_recipients).shape[0])

    else:
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)


    # ratio of internal emails to external emails
    if len(sent_data) != 0:
        int_porportion, ext_porportion = get_int_ext_porportion(sent_data)
        features.append(int_porportion)
        features.append(ext_porportion)

        # portion of emails inside/outside working hours
        # sent_in_working_hrs = sum([dd[0].hour > 8 and dd[0].hour < 18 for dd in sent_data])
        # tmp = sent_in_working_hrs/len(sent_data)
        # features.append(tmp)
        # features.append(1-tmp)

    else:
        features.append(0)
        features.append(0)
        # features.append(0)
        # features.append(0)

    # len received,
    # 2. received features
    recieved_data = data['received']

    # number of received emails
    features.append(len(recieved_data))

    # min, max, mean, std size of emails
    if len(recieved_data) != 0:
        sizes = np.array([dd[2] for dd in recieved_data])
        features.append(sizes.min())
        features.append(sizes.max())
        features.append(sizes.mean())
        features.append(sizes.std())

        # number of unique senders
        senders = [dd[1] for dd in recieved_data]
        features.append(len(set(senders)))

        from_telstra = ['telstra' in ss for ss in senders]
        tmp = sum(from_telstra)/len(senders)
        features.append(tmp)
        features.append(1-tmp)

    else:
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)


    return features

def train_model(X, beta, tau):

    pca = PCA()
    pca.fit(X)

    K = (pca.explained_variance_ratio_.cumsum() < tau).sum()
    U = pca.components_[:, :K]
    P = np.eye(U.shape[0]) - np.dot(U, U.T)

    theta1 = np.power(pca.explained_variance_[K:], 1).sum()
    theta2 = np.power(pca.explained_variance_[K:], 2).sum()
    theta3 = np.power(pca.explained_variance_[K:], 3).sum()
    h0 = 1 - (2 * theta1 * theta3) / (3 * theta2**2)
    c_beta = scipy.stats.norm.ppf(1 - beta)

    E1 = c_beta * np.sqrt(2 * theta2 * h0**2) / theta1
    E2 = theta2 * h0 * (h0 - 1) / (theta1 ** 2)
    Q = theta1 * ((E1 + 1 + E2) ** (1/h0))

    SPE = np.linalg.norm(np.dot(X, P), axis=1)

    return SPE, Q

def get_user_features(user, user_send_data, user_receive_data):

    features = []
    d0 = pd.to_datetime('2017-07-01')
    for i in range(31):
        date = d0 + i* pd.Timedelta('1 day')

        date_data = {}
        date_data['sent'] = get_date_data(user_send_data, date)
        date_data['received'] = get_date_data(user_receive_data, date)

        features.append(get_features(date_data))

    features = np.array(features)
    return features


def count_anomalies(SPE, Q, window=4):
    is_anomaly = SPE>Q
    n_windows = len(SPE) // (2*window) + 1

    n_anomalies = []
    start = 0
    for i in range(n_windows):
        end = start + 2*window
        n_anomalies.append((is_anomaly[start:end]).sum())
        start = end

    return n_anomalies

    
def main():

    """
    # path to input data directory
    pin = '/home/adhamb/ironport_data'

    # get file names
    sender_file_names = find_files(pin, 'sender')
    recipient_file_names = find_files(pin, 'recipient')

    # construct sender and recipient dictionaries
    sender_data = defaultdict(list)
    for file_name in tqdm(sender_file_names[:31], desc='sender data'):
        data = pd.read_pickle(file_name)
        for k, v in data.items():
            sender_data[k] += v

    recipient_data = defaultdict(list)
    for file_name in tqdm(recipient_file_names[:31], desc='recipient data'):
        data = pd.read_pickle(file_name)
        for k, v in data.items():
            recipient_data[k] += v

    # only consider emails that have sent and recieved messages.
    # This potentially gets rid of all automatic emails too.
    sender_users = list(sender_data.keys())
    recipient_users = list(recipient_data.keys())
    users = set(sender_users).intersection(set(recipient_users))
    users = np.array(list(users))

    # choose a subset of users that send and recieve between 100-200 emails
    sent_received_count = [
        (u, len(sender_data[u]), len(recipient_data[u])) for u in users
    ]
    sent_counts = np.array([src[1] for src in sent_received_count])
    received_counts = np.array([src[2] for src in sent_received_count])

    idxs = (received_counts>100) & (received_counts<200) & (sent_counts>100) & (sent_counts<200)
    users = users[idxs]

    sender_data2 = defaultdict(list)
    for user in users:
        sender_data2[user] = sender_data[user]
    recipient_data2 = defaultdict(list)
    for user in users:
        recipient_data2[user] = recipient_data[user]

    pd.to_pickle(
        {'sender_data': sender_data2, 'recipient_data': recipient_data2},
        'parsed_data.pkl'
    )
    """
    data = pd.read_pickle('parsed_data.pkl')
    users = data['sender_data'].keys()

    user = 'Joseph.Ip@team.telstra.com'
    user_send_data = data['sender_data'][user]
    user_receive_data = data['recipient_data'][user]
    features = get_user_features(user, user_send_data, user_receive_data)
    X = scale(features)
    SPE, Q = train_model(X, BETA, TAU)
    print(Q)
    print(SPE)
    1/0




if __name__ == '__main__':
    main()
