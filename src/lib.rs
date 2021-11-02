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

use core::{
    fmt::Debug,
    iter::{FilterMap, FusedIterator},
};

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

    /// Returns the contained value if it's `Ok`, otherwise, it calls `f` and forwards
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
            $crate::NullableResult::Ok(item) => item,
            $crate::NullableResult::Err(err) => {
                return NullableResult::Err(err.into());
            }
            $crate::NullableResult::None => {
                return NullableResult::None;
            }
        }
    }};
}

pub trait GeneralIterExt<T, E> {
    fn try_find<P>(self, pred: P) -> NullableResult<T, E>
    where
        P: FnMut(&T) -> Result<bool, E>;

    fn try_find_map<F, U>(self, f: F) -> NullableResult<U, E>
    where
        F: FnMut(T) -> NullableResult<U, E>;
}

impl<T, E, I: Iterator<Item = T>> GeneralIterExt<T, E> for I {
    #[inline]
    fn try_find<P>(mut self, mut pred: P) -> NullableResult<T, E>
    where
        P: FnMut(&T) -> Result<bool, E>,
    {
        use NullableResult::*;
        while let Some(item) = self.next() {
            return match pred(&item) {
                Result::Err(err) => Err(err),
                Result::Ok(true) => Ok(item),
                Result::Ok(false) => continue,
            };
        }
        None
    }

    #[inline]
    fn try_find_map<F, U>(mut self, mut f: F) -> NullableResult<U, E>
    where
        F: FnMut(T) -> NullableResult<U, E>,
    {
        use NullableResult::*;
        while let Some(item) = self.next() {
            return match f(item) {
                Ok(item) => Ok(item),
                Err(err) => Err(err),
                None => continue,
            };
        }
        None
    }
}

pub trait IterExt<T, E>: Iterator<Item = NullableResult<T, E>>
where
    Self: Sized,
{
    #[inline]
    fn filter_nulls(self) -> FilterNulls<Self, T, E> {
        self.filter_map(Option::from)
    }

    #[inline]
    fn extract_and_find<P>(self, mut pred: P) -> NullableResult<T, E>
    where
        P: FnMut(&T) -> Result<bool, E>,
    {
        use NullableResult::*;
        self.try_find_map(|item| {
            let item = extract!(item);
            match pred(&item) {
                Result::Err(err) => Err(err),
                Result::Ok(true) => Ok(item),
                Result::Ok(false) => None,
            }
        })
    }

    #[inline]
    fn extract_and_find_map<F, U>(self, mut f: F) -> NullableResult<U, E>
    where
        F: FnMut(T) -> NullableResult<U, E>,
    {
        self.try_find_map(|item| f(extract!(item)))
    }

    #[inline]
    fn try_filter<P>(self, pred: P) -> TryFilter<Self, P, T, E>
    where
        P: FnMut(&T) -> bool,
    {
        TryFilter { inner: self, pred }
    }

    #[inline]
    fn try_filter_map<F, U>(self, f: F) -> TryFilterMap<Self, F, T, U, E>
    where
        F: FnMut(T) -> Option<NullableResult<U, E>>,
    {
        TryFilterMap { inner: self, f }
    }
}

impl<I, T, E> IterExt<T, E> for I where I: Iterator<Item = NullableResult<T, E>> {}

type FilterNulls<I, T, E> =
    FilterMap<I, fn(NullableResult<T, E>) -> Option<Result<T, E>>>;

pub struct TryFilter<I, P, T, E>
where
    I: Iterator<Item = NullableResult<T, E>>,
    P: FnMut(&T) -> bool,
{
    inner: I,
    pred: P,
}

impl<I, P, T, E> Iterator for TryFilter<I, P, T, E>
where
    I: Iterator<Item = NullableResult<T, E>>,
    P: FnMut(&T) -> bool,
{
    type Item = NullableResult<T, E>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next() {
            None => None,
            Some(NullableResult::None) => Some(NullableResult::None),
            Some(NullableResult::Err(err)) => Some(NullableResult::Err(err)),
            Some(NullableResult::Ok(item)) if (self.pred)(&item) => {
                Some(NullableResult::Ok(item))
            }
            Some(NullableResult::Ok(_)) => self.next(),
        }
    }
}

impl<I, P, T, E> FusedIterator for TryFilter<I, P, T, E>
where
    I: FusedIterator<Item = NullableResult<T, E>>,
    P: FnMut(&T) -> bool,
{
}

pub struct TryFilterMap<I, F, T, U, E>
where
    I: Iterator<Item = NullableResult<T, E>>,
    F: FnMut(T) -> Option<NullableResult<U, E>>,
{
    inner: I,
    f: F,
}

impl<I, F, T, U, E> Iterator for TryFilterMap<I, F, T, U, E>
where
    I: Iterator<Item = NullableResult<T, E>>,
    F: FnMut(T) -> Option<NullableResult<U, E>>,
{
    type Item = NullableResult<U, E>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next() {
            None => None,
            Some(NullableResult::None) => Some(NullableResult::None),
            Some(NullableResult::Err(err)) => Some(NullableResult::Err(err)),
            Some(NullableResult::Ok(item)) => (self.f)(item),
        }
    }
}

impl<I, F, T, U, E> FusedIterator for TryFilterMap<I, F, T, U, E>
where
    I: FusedIterator<Item = NullableResult<T, E>>,
    F: FnMut(T) -> Option<NullableResult<U, E>>,
{
}
