#![cfg_attr(not(feature = "std"), no_std)]

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum NullableResult<T, E> {
    Ok(T),
    Err(E),
    None,
}

impl<T, E> From<Result<Option<T>, E>> for NullableResult<T, E> {
    #[inline]
    fn from(res: Result<Option<T>, E>) -> Self {
        match res {
            Ok(Some(item)) => NullableResult::Ok(item),
            Ok(None) => NullableResult::None,
            Err(err) => NullableResult::Err(err),
        }
    }
}

impl<T, E> From<NullableResult<T, E>> for Result<Option<T>, E> {
    #[inline]
    fn from(nr: NullableResult<T, E>) -> Self {
        match nr {
            NullableResult::Ok(item) => Ok(Some(item)),
            NullableResult::Err(err) => Err(err),
            NullableResult::None => Ok(None),
        }
    }
}

impl<T, E> From<Option<Result<T, E>>> for NullableResult<T, E> {
    #[inline]
    fn from(opt: Option<Result<T, E>>) -> Self {
        match opt {
            None => NullableResult::None,
            Some(Ok(item)) => NullableResult::Ok(item),
            Some(Err(err)) => NullableResult::Err(err),
        }
    }
}

impl<T, E> From<NullableResult<T, E>> for Option<Result<T, E>> {
    #[inline]
    fn from(nr: NullableResult<T, E>) -> Self {
        match nr {
            NullableResult::Ok(item) => Some(Ok(item)),
            NullableResult::Err(err) => Some(Err(err)),
            NullableResult::None => None,
        }
    }
}
