use std::convert::Infallible;
use std::fmt::Display;

/// Ckmeans Errors
#[derive(Debug)]
pub enum CkmeansErr {
    TooFewClassesError,
    TooManyClassesError,
    ConversionError,
    LowWindowError,
    HighWindowError,
    InfallibleError,
    InvalidRangeError,
}

impl Display for CkmeansErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CkmeansErr::ConversionError => {
                write!(f, "An error occurred during numeric conversion")
            }
            CkmeansErr::LowWindowError => {
                write!(f, "Couldn't get last element of low window")
            }
            CkmeansErr::HighWindowError => {
                write!(f, "Couldn't get first element of high window")
            }
            CkmeansErr::TooFewClassesError => {
                write!(f, "You can't specify 0 classes. Try a positive number")
            }
            CkmeansErr::TooManyClassesError => {
                write!(
                    f,
                    "You can't generate more classes than there are data values"
                )
            }
            CkmeansErr::InfallibleError => {
                write!(f, "An infallible numeric conversion failed")
            }
            CkmeansErr::InvalidRangeError => {
                write!(f, "k_min must be less than or equal to k_max")
            }
        }
    }
}

impl From<Infallible> for CkmeansErr {
    fn from(_: Infallible) -> Self {
        CkmeansErr::InfallibleError
    }
}

impl std::error::Error for CkmeansErr {}
