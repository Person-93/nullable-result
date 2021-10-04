//! A replacement for `Option<Result<T, E>>` or `Result<Option<T>, E>`.
//!
//! Sometimes, no value returned from an operation is not an error. It's a special
//! case that needs to be handled, but it's separate from error handling. Wrapping
//! an `Option` in a `Result` or vice versa can get very confusing very fast. Instead,
//! use a `NullableResult`.
//!
//! This is how it's defined:
//! ```rust
//! pub enum NullableResult<T, E> {
//!     Ok(T),
//!     Err(E),
//!     None,
//! }
//! ```
//!
//! ## Convert to and From std Types
//!
//! It defines the `From` trait for `Option<Result<T, E>>` and for
//! `Result<Option<T>, E>` in both directions, so you can easily convert between the
//! standard library types and back.
//! ```rust
//! use nullable_result::NullableResult;
//!
//! let opt_res: Option<Result<usize, isize>> = Some(Ok(4));
//! let nr: NullableResult<usize, isize> = NullableResult::from(opt_res);
//! let opt_res: Option<Result<_,_>> = nr.into();
//! ```
//! Normally, you don't need to annotate the types so much, but in this example,
//! there's not enough info for the compiler to infer the types.
//!
//! ## Unwrapping Conversions
//! There are also methods for unwrapping conversions. (i.e. convert to `Option<T>` or
//! `Result<T, E>`. When converting a a `Result`, you need to provide an error value
//! in case the `NullableResult` contains `None`.
//! ```rust
//! use nullable_result::NullableResult;
//! let nr = NullableResult::<usize, isize>::Ok(5);
//! let opt: Option<usize> = nr.clone().option();
//! let res: Result<usize, isize> = nr.result(-5);
//! ```
//!
//! ## Extract the Value
//! The crate comes with a convenience macro `extract` that works like the `?` operator
//! and cna be used in functions that return a `NullableResult` as long as the error
//! type is the same. It takes a `NullableResult`, if it contains an `Ok` value, the
//! value is extracted and returned, if it contains an `Err` or `None`, the function
//! returns early with the `Err` or `None` wrapped in a new `NullableResult`.
//! ```rust
//! use nullable_result::{NullableResult, extract};
//!
//! fn do_a_thing() -> NullableResult<usize, isize> {
//!     let res = some_other_func();
//!     let number = extract!(res); // <---- this will cause the function to return early
//!     NullableResult::Ok(number as usize + 5)
//! }
//!
//! // note that the two functions have different types for their Ok values
//! fn some_other_func() -> NullableResult<i8, isize> {
//!     NullableResult::None
//! }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

use core::{fmt::Debug, iter::FilterMap};

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
#[must_use]
pub enum NullableResult<T, E> {
    Ok(T),
    Err(E),
    None,
}

impl<T, E> Default for NullableResult<T, E> {
    fn default() -> Self {
        NullableResult::None
    }
}

impl<T, E: Debug> NullableResult<T, E> {
    /// Panics if it's not `Ok`, otherwise returns the contained value.
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
    /// Returns the contained value if it's `Ok`, returns `item` otherwise.
    #[inline]
    pub fn unwrap_or(self, item: T) -> T {
        match self {
            NullableResult::Ok(item) => item,
            _ => item,
        }
    }

    /// Returns the contained value if it's `Ok`, otherwise, it calls `f` and fowards
    /// its return value.
    #[inline]
    pub fn unwrap_or_else<F: FnOnce() -> T>(self, f: F) -> T {
        match self {
            NullableResult::Ok(item) => item,
            _ => f(),
        }
    }

    /// Returns an `Option<T>` consuming `self`, returns `None` if the `NullableResult`
    /// contains `Err`.
    #[inline]
    pub fn option(self) -> Option<T> {
        match self {
            NullableResult::Ok(item) => Some(item),
            NullableResult::Err(_) | NullableResult::None => None,
        }
    }

    /// Returns a `Result<T, E>`, returns the provided `err` if the `NullableResult`
    /// contains `None`
    #[inline]
    pub fn result(self, err: E) -> Result<T, E> {
        match self {
            NullableResult::Ok(item) => Ok(item),
            NullableResult::Err(err) => Err(err),
            NullableResult::None => Err(err),
        }
    }

    /// Returns a `Result<T, E>`, if the `NullableResult` contains `Ok` or `Err`, the
    /// value is returned, otherwise, returns the result of `f`.
    #[inline]
    pub fn result_with<F: FnOnce() -> E>(self, f: F) -> Result<T, E> {
        match self {
            NullableResult::Ok(item) => Ok(item),
            NullableResult::Err(err) => Err(err),
            NullableResult::None => Err(f()),
        }
    }

    /// Maps to a `NullableResult` with a different ok type.
    #[inline]
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> NullableResult<U, E> {
        match self {
            NullableResult::Ok(item) => NullableResult::Ok(f(item)),
            NullableResult::Err(err) => NullableResult::Err(err),
            NullableResult::None => NullableResult::None,
        }
    }

    /// Maps to a `NullableResult` with a different err type.
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

impl<T, E> From<Result<T, E>> for NullableResult<T, E> {
    #[inline]
    fn from(res: Result<T, E>) -> Self {
        match res {
            Ok(item) => NullableResult::Ok(item),
            Err(err) => NullableResult::Err(err),
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

impl<T, E> From<Option<T>> for NullableResult<T, E> {
    #[inline]
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(item) => NullableResult::Ok(item),
            None => NullableResult::None,
        }
    }
}

#[macro_export]
macro_rules! extract {
    ($nr:expr) => {{
        let nr = NullableResult::from($nr);
        match nr {
            nullable_result::NullableResult::Ok(item) => item,
            nullable_result::NullableResult::Err(err) => {
                return NullableResult::Err(err);
            }
            nullable_result::NullableResult::None => {
                return NullableResult::None;
            }
        }
    }};
}

pub trait IterExt<T, E>: Iterator<Item = NullableResult<T, E>>
where
    Self: Sized,
{
    fn filter_nulls(self) -> FilterNulls<Self, T, E> {
        self.filter_map(Option::from)
    }
}

impl<I, T, E> IterExt<T, E> for I where I: Iterator<Item = NullableResult<T, E>> {}

type FilterNulls<I, T, E> =
    FilterMap<I, fn(NullableResult<T, E>) -> Option<Result<T, E>>>;
