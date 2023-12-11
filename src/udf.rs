use std::sync::Arc;

use pyo3::{prelude::*, types::PyTuple};

use datafusion::arrow::array::{make_array, Array, ArrayData, ArrayRef};
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::pyarrow::{FromPyArrow, PyArrowType, ToPyArrow};
use datafusion::error::DataFusionError;
use datafusion::physical_plan::functions::make_scalar_function;
use datafusion::physical_plan::udf::ScalarUDF;
use datafusion_expr::create_udf;
use datafusion_expr::function::ScalarFunctionImplementation;

use crate::utils::parse_volatility;

/// Create a DataFusion's UDF implementation from a python function
/// that expects pyarrow arrays. This is more efficient as it performs
/// a zero-copy of the contents.
fn to_rust_function(func: PyObject) -> ScalarFunctionImplementation {
    make_scalar_function(
        move |args: &[ArrayRef]| -> Result<ArrayRef, DataFusionError> {
            Python::with_gil(|py| {
                // 1. cast args to Pyarrow arrays
                let py_args = args
                    .iter()
                    .map(|arg| arg.into_data().to_pyarrow(py).unwrap())
                    .collect::<Vec<_>>();
                let py_args = PyTuple::new(py, py_args);

                // 2. call function
                let value = func
                    .as_ref(py)
                    .call(py_args, None)
                    .map_err(|e| DataFusionError::Execution(format!("{e:?}")))?;

                // 3. cast to arrow::array::Array
                let array_data = ArrayData::from_pyarrow(value).unwrap();
                Ok(make_array(array_data))
            })
        },
    )
}

/// Represents a PyScalarUDF
#[pyclass(name = "ScalarUDF", module = "datafusion", subclass)]
#[derive(Debug, Clone)]
pub struct PyScalarUDF {
    pub(crate) function: ScalarUDF,
}

#[pymethods]
impl PyScalarUDF {
    #[new]
    #[pyo3(signature=(name, func, input_types, return_type, volatility))]
    fn new(
        name: &str,
        func: PyObject,
        input_types: PyArrowType<Vec<DataType>>,
        return_type: PyArrowType<DataType>,
        volatility: &str,
    ) -> PyResult<Self> {
        let function = create_udf(
            name,
            input_types.0,
            Arc::new(return_type.0),
            parse_volatility(volatility)?,
            to_rust_function(func),
        );
        Ok(Self { function })
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("ScalarUDF({})", self.function.name))
    }
}
