from typing import List

from continuiti.data import OperatorDataset
from continuiti.operators.shape import OperatorShapes
from continuiti.operators import Operator


def get_shape_mismatches(
    operators: List[Operator], datasets: List[OperatorDataset]
) -> List[OperatorShapes]:
    """Evaluates if an operator outputs the same shape as a list of datasets expects.

    The list of operators needs to be initialized in the same order as the list of operator-datasets. For each matching
    operator and dataset, a single batched forward pass is performed. If the shape of the output does not match the
    expected ground truth, its shape is added to a list. If there are AssertionErrors or RuntimeErrors, the shape of
    this specific dataset is also added to the list. The list of failed shapes is returned.

    Args:
        operators: List of operator instances matching the list of datasets.
        datasets: list of operator datasets on which this property should be checked.

    Returns:
        List of `DatasetShapes` for which the evaluation failed.
    """
    failed_shapes = []
    for operator, dataset in zip(operators, datasets):
        x, u, y, v = dataset[:9]  # batch of size 9
        try:
            output = operator(x, u, y)
            if output.shape != v.shape:
                failed_shapes.append(dataset.shapes)
        except AssertionError or RuntimeError:
            failed_shapes.append(dataset.shapes)
    return failed_shapes
