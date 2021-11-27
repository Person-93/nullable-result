//! ## Contents
//! * [NullableResult] - the core of this crate
//! * the [extract] macro - early return from functions `?`-style
//! * [iterator extension](IterExt) - additional methods for iterators over [NullableResult]
//! * [general iterator extension](GeneralIterExt) - additional methods for all iterators
//! * [MaybeTryFrom] and [MaybeTryInto] - analogues of [TryFrom] and [TryInto]

#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

use self::NullableResult::*;
#[cfg(doc)]
use core::convert::TryInto;
use core::{
    convert::TryFrom,
    fmt::Debug,
    iter::{FilterMap, FromIterator, FusedIterator},
    ops::Deref,
};

/// A replacement for `Option<Result<T, E>>` or `Result<Option<T>, E>`.
///
/// Sometimes, no value returned from an operation is not an error. It's a special
/// case that needs to be handled, but it's separate from error handling. Wrapping
/// an [`Option`] in a [`Result`] or vice versa can get very confusing very fast. Instead,
/// use a [`NullableResult`].
///
/// This is how it's defined:
/// ```rust
/// pub enum NullableResult<T, E> {
///     Ok(T),
///     Err(E),
///     Null,
/// }
/// ```
///
/// ## Convert to and From std Types
///
/// It defines the [`From`] trait for `Option<Result<T, E>>` and for
/// `Result<Option<T>, E>` in both directions, so you can easily convert between the
/// standard library types and back.
/// ```rust
/// # use nullable_result::NullableResult;
/// let opt_res: Option<Result<usize, isize>> = Some(Ok(4));
/// let nr: NullableResult<usize, isize> = NullableResult::from(opt_res);
/// let opt_res: Option<Result<_,_>> = nr.into();
/// ```
/// It also defines [`From`] for [`Option<T>] and for [`Result<T, E>`].
/// ```rust
/// # use nullable_result::NullableResult;
/// let nr: NullableResult<_, isize> = NullableResult::from(Some(4));
/// let result: Result<usize, isize> = Ok(4);
/// let nr = NullableResult::from(result);
/// ```
///
/// ## Unwrapping Conversions
/// There are also methods for unwrapping conversions. (i.e. convert to `Option<T>` or
/// `Result<T, E>`. When converting a a `Result`, you need to provide an error value
/// in case the `NullableResult` contains `Null`.
/// ```rust
/// # use nullable_result::NullableResult;
/// let nr = NullableResult::<usize, isize>::Ok(5);
/// let opt = nr.option();
/// let res = nr.result(-5);
/// ```
///
/// There are also a few convenience methods.
/// ```rust
/// # use nullable_result::NullableResult;
/// let nr: NullableResult<usize, isize> = NullableResult::Ok(4);
/// let _ : Option<Result<usize, isize>> = nr.optional_result();
/// let _: Result<Option<usize>, isize> = nr.resulting_option();
/// let _: Result<usize, Option<isize>> = nr.result_optional_err();
/// let _: Result<usize, isize> = nr.result(-5); // need to provide default error value
/// let _: Result<usize, isize> = nr.result_with(|| 5 - 10); //closure that returns a default error
/// ```
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
#[must_use]
pub enum NullableResult<T, E> {
    /// Contains the success value
    Ok(T),
    /// Contains the error value
    Err(E),
    /// No value
    Null,
}

impl<T, E> Default for NullableResult<T, E> {
    /// The default value is `Null`
    fn default() -> Self {
        Null
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
            NullableResult::Null => {
                panic!("tried to unwrap a nullable result containing `Null`")
            }
        }
    }
}

impl<T: Default, E> NullableResult<T, E> {
    /// Returns the contained value if it's `Ok`, otherwise returns the default value
    /// for that type.
    #[inline]
    pub fn unwrap_or_default(self) -> T {
        match self {
            Ok(item) => item,
            _ => T::default(),
        }
    }
}

impl<T: Copy, E> NullableResult<&'_ T, E> {
    /// Returns a `NullableResult` with the `Ok` part copied.
    #[inline]
    pub fn copied(self) -> NullableResult<T, E> {
        self.map(|&item| item)
    }
}

impl<T: Copy, E> NullableResult<&'_ mut T, E> {
    /// Returns a `NullableResult` with the `Ok` part copied.
    #[inline]
    pub fn copied(self) -> NullableResult<T, E> {
        self.map(|&mut item| item)
    }
}

impl<T: Clone, E> NullableResult<&'_ T, E> {
    /// Returns a `NullableResult` with the `Ok` part cloned.
    #[inline]
    pub fn cloned(self) -> NullableResult<T, E> {
        self.map(|item| item.clone())
    }
}

impl<T: Clone, E> NullableResult<&'_ mut T, E> {
    /// Returns a `NullableResult` with the `Ok` part cloned.
    #[inline]
    pub fn cloned(self) -> NullableResult<T, E> {
        self.map(|item| item.clone())
    }
}

impl<T: Deref, E> NullableResult<T, E> {
    /// Coerce the `Ok` variant of the original result with `Deref` and returns the
    /// new `NullableResult`
    #[inline]
    pub fn as_deref(&self) -> NullableResult<&T::Target, &E> {
        match self {
            Ok(item) => Ok(item.deref()),
            Err(err) => Err(err),
            Null => Null,
        }
    }
}

impl<T, E> NullableResult<T, E> {
    /// Returns `true` if this result is an [`Ok`] value
    #[inline]
    #[must_use]
    pub fn is_ok(&self) -> bool {
        matches!(self, Ok(_))
    }

    /// Returns `true` if this result is an [`Err`] value
    #[inline]
    #[must_use]
    pub fn is_err(&self) -> bool {
        matches!(self, Err(_))
    }

    /// Returns `true` if this result is a [`Null`] value
    #[inline]
    #[must_use]
    pub fn is_null(&self) -> bool {
        matches!(self, Null)
    }

    /// Returns the contained [`Ok`] value, consuming `self`.
    ///
    /// # Panics
    /// Panics if the value is not [`Ok`] with the provided message.
    #[inline]
    #[track_caller]
    pub fn expect(self, msg: &str) -> T {
        match self {
            Ok(item) => item,
            _ => panic!("{}", msg),
        }
    }

    /// Returns the contained value if it's `Ok`, returns `item` otherwise.
    #[inline]
    pub fn unwrap_or(self, item: T) -> T {
        match self {
            Ok(item) => item,
            _ => item,
        }
    }

    /// Returns the contained value if it's `Ok`, otherwise, it calls `f` and forwards
    /// its return value.
    #[inline]
    pub fn unwrap_or_else<F: FnOnce() -> T>(self, f: F) -> T {
        match self {
            Ok(item) => item,
            _ => f(),
        }
    }

    /// Returns an `Option<T>` consuming `self`, returns `Null` if the `NullableResult`
    /// contains `Err`.
    #[inline]
    pub fn option(self) -> Option<T> {
        match self {
            Ok(item) => Some(item),
            Err(_) | Null => None,
        }
    }

    /// Return an `Option<Result<T, E>>` consuming `self`.
    #[inline]
    pub fn optional_result(self) -> Option<Result<T, E>> {
        self.into()
    }

    /// Return a `Result<Option<T>, E>` consuming `self`.
    #[inline]
    pub fn resulting_option(self) -> Result<Option<T>, E> {
        self.into()
    }

    /// Returns a `Result<T, E>`, returns the provided `err` if the `NullableResult`
    /// contains `Null`
    #[inline]
    pub fn result(self, err: E) -> Result<T, E> {
        match self {
            Ok(item) => Result::Ok(item),
            Err(err) => Result::Err(err),
            Null => Result::Err(err),
        }
    }

    /// Returns a `Result<T, E>`, if the `NullableResult` contains `Ok` or `Err`, the
    /// value is returned, otherwise, returns the result of `f`.
    #[inline]
    pub fn result_with<F: FnOnce() -> E>(self, f: F) -> Result<T, E> {
        match self {
            Ok(item) => Result::Ok(item),
            Err(err) => Result::Err(err),
            Null => Result::Err(f()),
        }
    }

    /// Maps to a `NullableResult` with a different ok type.
    #[inline]
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> NullableResult<U, E> {
        match self {
            Ok(item) => Ok(f(item)),
            Err(err) => Err(err),
            Null => Null,
        }
    }

    /// Maps to a `NullableResult` with a different err type.
    #[inline]
    pub fn map_err<U, F: FnOnce(E) -> U>(self, f: F) -> NullableResult<T, U> {
        match self {
            Ok(item) => NullableResult::Ok(item),
            Err(err) => NullableResult::Err(f(err)),
            Null => NullableResult::Null,
        }
    }

    /// Returns a [`Result`] with an optional error.
    #[inline]
    pub fn result_optional_err(self) -> Result<T, Option<E>> {
        match self {
            Ok(item) => Result::Ok(item),
            Err(err) => Result::Err(Some(err)),
            Null => Result::Err(None),
        }
    }

    /// Convert from a `NullableResult<T, E>` or `&NullableResult<T, E>` to a
    /// `NullableResult<&T, &E>`.
    #[inline]
    pub fn as_ref(&self) -> NullableResult<&T, &E> {
        use NullableResult::*;
        match self {
            Ok(item) => Ok(item),
            Err(err) => Err(err),
            Null => Null,
        }
    }

    /// Convert from a `mut NullableResult<T, E>` or `&mut NullableResult<T, E>` to a
    /// `NullableResult<&mut T, &mut E>`.
    #[inline]
    pub fn as_mut(&mut self) -> NullableResult<&mut T, &mut E> {
        use NullableResult::*;
        match self {
            Ok(item) => Ok(item),
            Err(err) => Err(err),
            Null => Null,
        }
    }

    /// If `self` is [`Ok`], returns `res`, keeps the [`Err`] or [`Null`] from
    /// `self` otherwise.
    #[inline]
    pub fn and<U>(self, res: NullableResult<U, E>) -> NullableResult<U, E> {
        match self {
            Ok(_) => res,
            Err(err) => Err(err),
            Null => Null,
        }
    }

    /// Calls `op` if the result is [`Ok`], otherwise returns the [`Err`] or [`Null`]
    /// from `self`.
    #[inline]
    pub fn and_then<U, F>(self, op: F) -> NullableResult<U, E>
    where
        F: FnOnce(T) -> NullableResult<U, E>,
    {
        match self {
            Ok(item) => op(item),
            Err(err) => Err(err),
            Null => Null,
        }
    }
}

impl<T, E> NullableResult<NullableResult<T, E>, E> {
    /// Convert from `NullableResult<NullableResult<T, E>, E>` to
    /// `NullableResult<T, E>`.
    #[inline]
    pub fn flatten(self) -> NullableResult<T, E> {
        match self {
            Ok(Ok(item)) => Ok(item),
            Ok(Err(err)) | Err(err) => Err(err),
            Ok(Null) | Null => Null,
        }
    }
}

impl<T, E> From<Result<Option<T>, E>> for NullableResult<T, E> {
    #[inline]
    fn from(res: Result<Option<T>, E>) -> Self {
        match res {
            Result::Ok(Option::Some(item)) => Ok(item),
            Result::Ok(None) => Null,
            Result::Err(err) => Err(err),
        }
    }
}

impl<T, E> From<NullableResult<T, E>> for Result<Option<T>, E> {
    #[inline]
    fn from(nr: NullableResult<T, E>) -> Self {
        match nr {
            Ok(item) => Result::Ok(Some(item)),
            Err(err) => Result::Err(err),
            Null => Result::Ok(None),
        }
    }
}

impl<T, E> From<Result<T, E>> for NullableResult<T, E> {
    #[inline]
    fn from(res: Result<T, E>) -> Self {
        match res {
            Result::Ok(item) => Ok(item),
            Result::Err(err) => Err(err),
        }
    }
}

impl<T, E> From<Option<Result<T, E>>> for NullableResult<T, E> {
    #[inline]
    fn from(opt: Option<Result<T, E>>) -> Self {
        match opt {
            None => Null,
            Some(Result::Ok(item)) => Ok(item),
            Some(Result::Err(err)) => Err(err),
        }
    }
}

impl<T, E> From<NullableResult<T, E>> for Option<Result<T, E>> {
    #[inline]
    fn from(nr: NullableResult<T, E>) -> Self {
        match nr {
            Ok(item) => Some(Result::Ok(item)),
            Err(err) => Some(Result::Err(err)),
            Null => None,
        }
    }
}

impl<T, E> From<Option<T>> for NullableResult<T, E> {
    #[inline]
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(item) => Ok(item),
            None => Null,
        }
    }
}

impl<T, E> From<Result<T, Option<E>>> for NullableResult<T, E> {
    #[inline]
    fn from(res: Result<T, Option<E>>) -> Self {
        match res {
            Result::Ok(item) => Ok(item),
            Result::Err(Some(err)) => Err(err),
            Result::Err(None) => Null,
        }
    }
}

impl<T, E, C> FromIterator<NullableResult<T, E>> for NullableResult<C, E>
where
    C: FromIterator<T>,
{
    fn from_iter<I: IntoIterator<Item = NullableResult<T, E>>>(
        iter: I,
    ) -> Self {
        let result = iter
            .into_iter()
            .map(NullableResult::result_optional_err)
            .collect::<Result<_, _>>();

        NullableResult::from(result)
    }
}

/// This macro [`extract`] that works like the `?` operator and can be used in
/// functions that return a [`NullableResult`] as long as the error type is the same.
/// It takes a [`NullableResult`], a [`Result`], or an [`Option`]. If the input
/// contains an `Ok` or `Some` value, the value is extracted and returned,
/// if it contains an `Err` or `Null`, the function returns early with the `Err` or
/// `Null` wrapped in a new `NullableResult`.
/// ```rust
/// # use nullable_result::{NullableResult, extract};
/// fn do_a_thing() -> NullableResult<usize, isize> {
///     let res = some_other_func();
///     let number = extract!(res); // <---- this will cause the function to return early
///     NullableResult::Ok(number as usize + 5)
/// }
///
/// // note that the two functions have different types for their Ok values
/// fn some_other_func() -> NullableResult<i8, isize> {
///     NullableResult::Null
/// }
/// ```
///
/// If the input is an option, it requires a second parameter as a type annotation for
/// converting to a ['NullableResult`]. Hopefully this won't be necessary in some
/// future version.
/// ```rust
/// # use nullable_result::{NullableResult, extract};
/// fn f() -> NullableResult<usize, isize> {
///     let opt = Some(4_usize);
///     let four = extract!(opt, isize);
///     NullableResult::Ok(four)
/// }
/// ```
#[macro_export]
macro_rules! extract {
    ($nr:expr) => {
        extract!($nr, _)
    };
    ($nr:expr, $err:ty) => {{
        let nr = $crate::NullableResult::<_, $err>::from($nr);
        match nr {
            $crate::NullableResult::Ok(item) => item,
            $crate::NullableResult::Err(err) => {
                return $crate::NullableResult::Err(err.into());
            }
            $crate::NullableResult::Null => {
                return $crate::NullableResult::Null;
            }
        }
    }};
}

/// Adds additional methods to all iterators.
pub trait GeneralIterExt: Iterator {
    /// Applies the predicate to the elements of the iterator and returns the first
    /// true result or the first error.
    fn try_find<E, P>(self, pred: P) -> NullableResult<Self::Item, E>
    where
        P: FnMut(&Self::Item) -> Result<bool, E>;

    /// Applies the function to the elements of the iterator and returns the first
    /// value that isn't [`Null`]
    fn try_find_map<T, E, F>(self, f: F) -> NullableResult<T, E>
    where
        F: FnMut(Self::Item) -> NullableResult<T, E>;

    /// Fold the elements of the iterator using the given initial value and operation.
    /// Returns early if the operation does not return [`Ok`].
    fn maybe_try_fold<T, E, Op>(
        &mut self,
        init: T,
        op: Op,
    ) -> NullableResult<T, E>
    where
        Op: FnMut(T, Self::Item) -> NullableResult<T, E>;
}

impl<I: Iterator> GeneralIterExt for I {
    #[inline]
    fn try_find<E, P>(self, mut pred: P) -> NullableResult<Self::Item, E>
    where
        P: FnMut(&Self::Item) -> Result<bool, E>,
    {
        for item in self {
            return match pred(&item) {
                Result::Err(err) => Err(err),
                Result::Ok(true) => Ok(item),
                Result::Ok(false) => continue,
            };
        }
        Null
    }

    #[inline]
    fn try_find_map<T, E, F>(self, mut f: F) -> NullableResult<T, E>
    where
        F: FnMut(Self::Item) -> NullableResult<T, E>,
    {
        for item in self {
            return match f(item) {
                Ok(item) => Ok(item),
                Err(err) => Err(err),
                Null => continue,
            };
        }
        Null
    }

    #[inline]
    fn maybe_try_fold<T, E, Op>(
        &mut self,
        init: T,
        mut op: Op,
    ) -> NullableResult<T, E>
    where
        Op: FnMut(T, Self::Item) -> NullableResult<T, E>,
    {
        self.try_fold(init, |prev, curr| op(prev, curr).result_optional_err())
            .into()
    }
}

/// Additional methods for iterators over [`NullableResult`]
pub trait IterExt<T, E>: Iterator<Item = NullableResult<T, E>>
where
    Self: Sized,
{
    /// Filter out all the null values. Returns an iterator over [`Result<T, E>`].
    #[inline]
    fn filter_nulls(self) -> FilterNulls<Self, T, E> {
        self.filter_map(Option::from)
    }

    /// Returns the first value that is an [`Err`] or that the predicate accepts.
    #[inline]
    fn extract_and_find<P>(self, mut pred: P) -> NullableResult<T, E>
    where
        P: FnMut(&T) -> Result<bool, E>,
    {
        self.try_find_map(|item| {
            let item = extract!(item);
            match pred(&item) {
                Result::Err(err) => Err(err),
                Result::Ok(true) => Ok(item),
                Result::Ok(false) => Null,
            }
        })
    }

    /// Applies the function to each element until it finds one that the function
    /// returns [`Ok`] or [`Err`] and returns that value.
    #[inline]
    fn extract_and_find_map<F, U>(self, mut f: F) -> NullableResult<U, E>
    where
        F: FnMut(T) -> NullableResult<U, E>,
    {
        self.try_find_map(|item| f(extract!(item)))
    }

    /// Returns an iterator that applies the predicate to each element and filters out
    /// the values for which it returns false.
    #[inline]
    fn try_filter<P>(self, pred: P) -> TryFilter<Self, P, T, E>
    where
        P: FnMut(&T) -> bool,
    {
        TryFilter { inner: self, pred }
    }

    /// Returns an iterator that both filters and maps.
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

/// See [IterExt::try_filter]
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

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next() {
            None => None,
            Some(Null) => Some(Null),
            Some(Err(err)) => Some(Err(err)),
            Some(Ok(item)) if (self.pred)(&item) => Some(Ok(item)),
            Some(Ok(_)) => self.next(),
        }
    }
}

impl<I, P, T, E> FusedIterator for TryFilter<I, P, T, E>
where
    I: FusedIterator<Item = NullableResult<T, E>>,
    P: FnMut(&T) -> bool,
{
}

/// See [IterExt::try_filter_map]
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
            Some(Null) => Some(Null),
            Some(Err(err)) => Some(Err(err)),
            Some(Ok(item)) => (self.f)(item),
        }
    }
}

impl<I, F, T, U, E> FusedIterator for TryFilterMap<I, F, T, U, E>
where
    I: FusedIterator<Item = NullableResult<T, E>>,
    F: FnMut(T) -> Option<NullableResult<U, E>>,
{
}

/// Analogue of [TryFrom] that returns a [`NullableResult`]
pub trait MaybeTryFrom<T>: Sized {
    /// The type that is returned if conversion fails
    type Error;

    /// Convert a `T` to [`NullableResult<Self, Self::Error>`]
    fn maybe_try_from(item: T) -> NullableResult<Self, Self::Error>;
}

/// Analogue of [TryInto] that returns a [`NullableResult`]
pub trait MaybeTryInto<T>: Sized {
    /// The type that is returned if conversion fails
    type Error;

    /// Convert a `Self` to [`NullableResult<T, Self::Error>`]
    fn maybe_try_into(self) -> NullableResult<T, Self::Error>;
}

impl<T, U: TryFrom<T>> MaybeTryFrom<T> for U {
    type Error = U::Error;

    #[inline]
    fn maybe_try_from(item: T) -> NullableResult<Self, Self::Error> {
        U::try_from(item).into()
    }
}

impl<T, U: MaybeTryFrom<T>> MaybeTryInto<U> for T {
    type Error = U::Error;

    #[inline]
    fn maybe_try_into(self) -> NullableResult<U, Self::Error> {
        U::maybe_try_from(self)
    }
}
