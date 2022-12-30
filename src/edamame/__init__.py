from .eda import (dimensions,
describe_distribution,
identify_types,
num_to_categorical,
missing,
handling_missing,
drop_columns,
plot_categorical,
plot_numerical,
view_cardinality,
modify_cardinality,
correlation_pearson,
correlation_categorical,
correlation_phik,
quant_variable_study,
interaction,
inspection,
split_and_scaling)

from .regression import (TrainRegressor, regression_metrics)

from .tools import load_model

__version__ = "0.23"
