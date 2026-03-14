from collections import OrderedDict
gru4rec_params = OrderedDict([
('loss', 'cross-entropy'),
('constrained_embedding', True),
('layers', [512]),
('n_epochs', 5),
('batch_size', 144),
('dropout_p_embed', 0.35),
('dropout_p_hidden', 0.0),
('learning_rate', 0.045),
('momentum', 0.3),
('n_sample', 2048),
('sample_alpha', 0.0),
('logq', 1.0)
])
