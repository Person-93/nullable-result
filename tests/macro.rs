#![cfg_attr(not(feature = "std"), no_std)]

use nullable_result::*;

#[test]
fn ok() {
    let res = helper(NullableResult::Ok(1), 1);
    assert!(matches!(res, NullableResult::Ok(_)));
}

#[test]
fn err() {
    let res = helper(NullableResult::Err(4), 4);
    assert!(matches!(res, NullableResult::Err(4)));
}

#[test]
fn none() {
    let res = helper(NullableResult::None, 0);
    assert!(matches!(res, NullableResult::None));
}

fn helper(nr: NullableResult<i32, u32>, n: i32) -> NullableResult<(), u32> {
    let item = extract!(nr);
    assert_eq!(item, n);
    NullableResult::Ok(())
}
