import cebra
import tempfile
from pathlib import Path
from numpy.random import uniform, randint
from sklearn.model_selection import train_test_split

from percephone.analysis.utils import get_zscore
from percephone.utils.io import get_recs_dict

recs = get_recs_dict("Célien")
rec = recs[4456]

# 1. Define a CEBRA model
cebra_model = cebra.CEBRA(
    model_architecture="offset10-model",
    batch_size=512,
    learning_rate=1e-4,
    max_iterations=10000,
    max_adapt_iterations=200, # TODO(user): to change to ~100-500
    time_offsets=10,
    output_dimension=8,
    verbose=False
)

# 2. Load example data
neural_data = get_zscore(rec, exc_neurons=True, inh_neurons=False, time_span="stim", window=0.5, estimator="Mean", sort=False, amp_sort=False)
# new_neural_data = cebra.load_data(file="neural_data.npz", key="new_neural")
# continuous_label = cebra.load_data(file="auxiliary_behavior_data.h5", key="auxiliary_variables", columns=["continuous1", "continuous2", "continuous3"])
discrete_label = rec.detected_stim

# assert neural_data.shape == (100, 3)
# assert new_neural_data.shape == (100, 4)
# assert discrete_label.shape == (100, )
# assert continuous_label.shape == (100, 3)

# 3. Split data and labels
(
    train_data,
    valid_data,
    train_discrete_label,
    valid_discrete_label,
    # train_continuous_label,
    # valid_continuous_label,
) = train_test_split(neural_data,
                    discrete_label,
                    # continuous_label,
                    test_size=0.3)

# 4. Fit the model
# time contrastive learning
cebra_model.fit(train_data)
# discrete behavior contrastive learning
cebra_model.fit(train_data, train_discrete_label,)
# continuous behavior contrastive learning
# cebra_model.fit(train_data, train_continuous_label)
# mixed behavior contrastive learning
# cebra_model.fit(train_data, train_discrete_label, train_continuous_label)

# 5. Save the model
tmp_file = Path(tempfile.gettempdir(), 'cebra.pt')
cebra_model.save(tmp_file)

# 6. Load the model and compute an embedding
cebra_model = cebra.CEBRA.load(tmp_file)
train_embedding = cebra_model.transform(train_data)
valid_embedding = cebra_model.transform(valid_data)
# assert train_embedding.shape == (70, 8)
# assert valid_embedding.shape == (30, 8)

# 7. Evaluate the model performances
goodness_of_fit = cebra.sklearn.metrics.infonce_loss(cebra_model,
                                                     valid_data,
                                                     valid_discrete_label,
                                                     # valid_continuous_label,
                                                     num_batches=5)

# 8. Adapt the model to a new session
# cebra_model.fit(new_neural_data, adapt = True)

# 9. Decode discrete labels behavior from the embedding
decoder = cebra.KNNDecoder()
decoder.fit(train_embedding, train_discrete_label)
prediction = decoder.predict(valid_embedding)
# assert prediction.shape == (30,)