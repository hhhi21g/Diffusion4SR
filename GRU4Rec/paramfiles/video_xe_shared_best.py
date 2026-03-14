from collections import OrderedDict
gru4rec_params = OrderedDict([
('loss', 'cross-entropy'),
('constrained_embedding', True),
('layers', [512]),
('n_epochs', 5),
('batch_size', 64),
('dropout_p_embed', 0.05),
('dropout_p_hidden', 0.0),
('learning_rate', 0.03),
('momentum', 0.65),
('n_sample', 2048),
('sample_alpha', 1.0),
('logq', 1.0)
])
