#![cfg_attr(not(feature = "std"), no_std)]

use core::fmt::Debug;

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
#[must_use]
pub enum NullableResult<T, E> {
    Ok(T),
    Err(E),
    None,
}

impl<T, E: Debug> NullableResult<T, E> {
    #[inline]
    pub fn unwrap(self) -> T {
        match self {
            NullableResult::Ok(item) => item,
            NullableResult::Err(err) => panic!(
                "tried to unwrap a nullable result containing Err: {:?}",
                err
            ),
            NullableResult::None => {
                panic!("tried to unwrap a nullable result containing None")
            }
        }
    }
}

impl<T, E> NullableResult<T, E> {
    #[inline]
    pub fn unwrap_or(self, item: T) -> T {
        match self {
            NullableResult::Ok(item) => item,
            _ => item,
        }
    }

    #[inline]
    pub fn unwrap_or_else<F: FnOnce() -> T>(self, f: F) -> T {
        match self {
            NullableResult::Ok(item) => item,
            _ => f(),
        }
    }

    #[inline]
    pub fn option(self) -> Option<T> {
        match self {
            NullableResult::Ok(item) => Some(item),
            NullableResult::Err(_) | NullableResult::None => None,
        }
    }

    #[inline]
    pub fn result(self, item: T) -> Result<T, E> {
        match self {
            NullableResult::Ok(item) => Ok(item),
            NullableResult::Err(err) => Err(err),
            NullableResult::None => Ok(item),
        }
    }

    #[inline]
    pub fn result_with<F: FnOnce() -> T>(self, f: F) -> Result<T, E> {
        match self {
            NullableResult::Ok(item) => Ok(item),
            NullableResult::Err(err) => Err(err),
            NullableResult::None => Ok(f()),
        }
    }

    #[inline]
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> NullableResult<U, E> {
        match self {
            NullableResult::Ok(item) => NullableResult::Ok(f(item)),
            NullableResult::Err(err) => NullableResult::Err(err),
            NullableResult::None => NullableResult::None,
        }
    }

    #[inline]
    pub fn map_err<U, F: FnOnce(E) -> U>(self, f: F) -> NullableResult<T, U> {
        match self {
            NullableResult::Ok(item) => NullableResult::Ok(item),
            NullableResult::Err(err) => NullableResult::Err(f(err)),
            NullableResult::None => NullableResult::None,
        }
    }
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
