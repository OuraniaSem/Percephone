import sys

import matplotlib.pyplot as plt
import cebra.data
import cebra.datasets
import cebra.integrations
from cebra import CEBRA
import matplotlib.pyplot as plt
import pickle
from percephone.analysis.utils import get_zscore
from percephone.cebra.CEBRA_trials_viz import get_recs_dict

recs = get_recs_dict("Célien")

hypo_only = False

names_wt, datas_wt, labels_wt = [], [], []
names_ko, datas_ko, labels_ko = [], [], []

for rec in recs.values():
    if not (rec.genotype == "KO" and hypo_only):
        name = rec.filename
        label = rec.detected_stim.astype(int)
        data = \
        get_zscore(rec, exc_neurons=True, inh_neurons=True, time_span="stim", window=0.5, estimator="Mean", sort=False,
                   amp_sort=False, single_frame_estimator=True)[0].T

        if rec.genotype == "WT":
            names_wt.append(name)
            datas_wt.append(data)
            labels_wt.append(label)
        else:
            names_ko.append(name)
            datas_ko.append(data)
            labels_ko.append(label)
    else:
        continue

print(f"WT: {names_wt}")
print(f"KO: {names_ko}")


# ========== Single Session ==========
max_iter=1000

embeddings_wt = dict()
embeddings_ko = dict()

# Single session training
for name, X, y in zip(names_wt, datas_wt, labels_wt):
    # Fit one CEBRA model per session (i.e., per rat)
    print(f"Fitting CEBRA for {name}")
    cebra_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=3,
                        max_iterations=max_iter,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

    cebra_model.fit(X, y)
    embeddings_wt[name] = cebra_model.transform(X)
for name, X, y in zip(names_ko, datas_ko, labels_ko):
    # Fit one CEBRA model per session (i.e., per rat)
    print(f"Fitting CEBRA for {name}")
    cebra_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=3,
                        max_iterations=max_iter,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

    cebra_model.fit(X, y)
    embeddings_ko[name] = cebra_model.transform(X)


# Align the single session embeddings to the first rat
alignment = cebra.data.helper.OrthogonalProcrustesAlignment()
first_rat_wt = list(embeddings_wt.keys())[0]
first_rat_ko = list(embeddings_ko.keys())[0]

for j, rat_name in enumerate(list(embeddings_wt.keys())[1:]):
    embeddings_wt[f"{rat_name}"] = alignment.fit_transform(
        embeddings_wt[first_rat_wt], embeddings_wt[rat_name], labels_wt[0], labels_wt[j+1])
for j, rat_name in enumerate(list(embeddings_ko.keys())[1:]):
    embeddings_ko[f"{rat_name}"] = alignment.fit_transform(
        embeddings_ko[first_rat_ko], embeddings_ko[rat_name], labels_ko[0], labels_ko[j+1])

# Save embeddings in current folder
with open('embeddings_wt.pkl', 'wb') as f:
    pickle.dump(embeddings_wt, f)
with open('embeddings_ko.pkl', 'wb') as f:
    pickle.dump(embeddings_ko, f)




# ========== Multi Session ==========
multi_embeddings_wt = dict()
multi_embeddings_ko = dict()

# Multisession training
multi_cebra_model_wt = CEBRA(model_architecture='offset10-model',
                    batch_size=512,
                    learning_rate=3e-4,
                    temperature=1,
                    output_dimension=3,
                    max_iterations=max_iter,
                    distance='cosine',
                    conditional='time_delta',
                    device='cuda_if_available',
                    verbose=True,
                    time_offsets=10)
multi_cebra_model_ko = CEBRA(model_architecture='offset10-model',
                    batch_size=512,
                    learning_rate=3e-4,
                    temperature=1,
                    output_dimension=3,
                    max_iterations=max_iter,
                    distance='cosine',
                    conditional='time_delta',
                    device='cuda_if_available',
                    verbose=True,
                    time_offsets=10)

# Provide a list of data, i.e. datas = [data_a, data_b, ...]
multi_cebra_model_wt.fit(datas_wt, labels_wt)
multi_cebra_model_ko.fit(datas_wt, labels_wt)

# Transform each session with the right model, by providing the corresponding session ID
for i, (name, X) in enumerate(zip(names_wt, datas_wt)):
    multi_embeddings_wt[name] = multi_cebra_model_wt.transform(X, session_id=i)
for i, (name, X) in enumerate(zip(names_ko, datas_ko)):
    multi_embeddings_ko[name] = multi_cebra_model_ko.transform(X, session_id=i)

# Save embeddings in current folder
with open('multi_embeddings_wt.pkl', 'wb') as f:
    pickle.dump(multi_embeddings_wt, f)
with open('multi_embeddings_ko.pkl', 'wb') as f:
    pickle.dump(multi_embeddings_ko, f)