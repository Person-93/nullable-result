use crate::NullableResult::{self, *};
use core::{
    convert::Infallible,
    ops::{ControlFlow, FromResidual, Try},
};

impl<T, E> Try for NullableResult<T, E> {
    type Output = T;
    type Residual = NullableResult<Infallible, E>;

    fn from_output(output: Self::Output) -> Self {
        Ok(output)
    }

    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        use ControlFlow::*;

        match self {
            Ok(item) => Continue(item),
            Err(err) => Break(Err(err)),
            Null => Break(Null),
        }
    }
}

impl<T, E1, E2> FromResidual<NullableResult<Infallible, E1>>
    for NullableResult<T, E2>
where
    E1: Into<E2>,
{
    fn from_residual(residual: NullableResult<Infallible, E1>) -> Self {
        match residual {
            Ok(_) => unreachable!(),
            Err(err) => Err(err.into()),
            Null => Null,
        }
    }
}

impl<T, E1, E2> FromResidual<Result<Infallible, E1>> for NullableResult<T, E2>
where
    E1: Into<E2>,
{
    fn from_residual(residual: Result<Infallible, E1>) -> Self {
        Err(residual.err().unwrap().into())
    }
}

impl<T, E> FromResidual<Option<Infallible>> for NullableResult<T, E> {
    fn from_residual(_: Option<Infallible>) -> Self {
        Null
    }
}
