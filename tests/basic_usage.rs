#![cfg_attr(not(feature = "std"), no_std)]

use nullable_result::NullableResult;

mod result_and_back {
    use super::*;

    #[test]
    fn ok() {
        let res: Result<Option<_>, u32> = Ok(Some(1));
        let nr: NullableResult<i32, u32> = NullableResult::from(res);
        assert!(matches!(nr, NullableResult::Ok(_)));
        let res = Result::from(nr);
        assert_eq!(res.unwrap().unwrap(), 1);
    }

    #[test]
    fn none() {
        let res = Ok(None);
        let nr: NullableResult<(), ()> = NullableResult::from(res);
        assert!(matches!(nr, NullableResult::None));
        let res = Result::from(nr);
        assert!(res.unwrap().is_none());
    }

    #[test]
    fn err() {
        let res: Result<Option<_>, _> = Err(1);
        let nr: NullableResult<(), _> = NullableResult::from(res);
        assert!(matches!(nr, NullableResult::Err(_)));
        let res = Result::from(nr);
        assert_eq!(res.unwrap_err(), 1);
    }
}

mod option_and_back {
    use super::*;

    #[test]
    fn ok() {
        let opt = Some(Ok(1));
        let nr: NullableResult<i32, ()> = NullableResult::from(opt);
        assert!(matches!(nr, NullableResult::Ok(_)));
        let opt: Option<Result<_, _>> = Option::from(nr);
        assert_eq!(opt.unwrap().unwrap(), 1);
    }

    #[test]
    fn none() {
        let opt = None;
        let nr: NullableResult<(), ()> = NullableResult::from(Ok(opt));
        assert!(matches!(nr, NullableResult::None));
        let opt: Option<Result<_, _>> = Option::from(nr);
        assert!(opt.is_none());
    }

    #[test]
    fn err() {
        let opt = Some(Err(1));
        let nr: NullableResult<(), _> = NullableResult::from(opt);
        assert!(matches!(nr, NullableResult::Err(_)));
        let opt: Option<Result<_, _>> = Option::from(nr);
        assert_eq!(opt.unwrap().unwrap_err(), 1);
    }
}
