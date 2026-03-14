from collections import OrderedDict
gru4rec_params = OrderedDict([
('loss', 'cross-entropy'),
('constrained_embedding', True),
('layers', [512]),
('n_epochs', 5),
('batch_size', 48),
('dropout_p_embed', 0.0),
('dropout_p_hidden', 0.05),
('learning_rate', 0.055),
('momentum', 0.4),
('n_sample', 2048),
('sample_alpha', 0.4),
('logq', 1.0)
])
