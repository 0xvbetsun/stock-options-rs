use std::f64::consts::{E, PI};

#[derive(Debug)]
pub enum MathError {
    NonPositiveStrike,
    NonPositiveStock,
    NonPositivePremium,
}

pub type MathResult = Result<f64, MathError>;

pub struct BlackScholesModel {
    opt: OptionKind,       // option type (call or put)
    strike: f64,           // strike price ($$$ per share)
    stock: f64,            // underlying price ($$$ per share)
    interest_rate: f64,    // continuously compounded risk-free interest rate (% p.a.)
    volatility: f64,       // volatility (% p.a.)
    time_to_expire: f64,   // time to expiration (% of year)
    dividend: Option<f64>, // continuously compounded dividend yield (% p.a.)
}

impl BlackScholesModel {
    pub fn new(
        opt: OptionKind,
        strike: f64,
        stock: f64,
        interest_rate: f64,
        volatility: f64,
        time_to_expire: f64,
        dividend: Option<f64>,
    ) -> BlackScholesModel {
        BlackScholesModel {
            opt,
            strike,
            stock,
            interest_rate,
            volatility,
            time_to_expire,
            dividend,
        }
    }
    pub fn price(&self) -> MathResult {
        let dividend = self.dividend.unwrap_or_default();

        let d1 = (self.stock / self.strike).ln() + self.interest_rate - dividend
            + self.volatility.powi(2) / 2.0 * self.time_to_expire / self.volatility
                * self.time_to_expire.sqrt();
        let d2 = d1 - self.volatility * self.time_to_expire.sqrt();

        match self.opt {
            OptionKind::Call => Ok(self.stock
                * E.powf(-dividend * self.time_to_expire)
                * norm_dist(d1)
                - self.strike * E.powf(-self.interest_rate * self.time_to_expire) * norm_dist(d2)),
            OptionKind::Put => Ok(self.strike
                * E.powf(-self.interest_rate * self.time_to_expire)
                * norm_dist(-d2)
                - self.stock * E.powf(-dividend * self.time_to_expire) * norm_dist(-d1)),
        }
    }
}

pub enum OptionKind {
    Call,
    Put,
}
pub enum Position {
    Long,
    Short,
}

// break_even_point calculates price in the underlying asset at which exercise/dispose
// of the contract without incurring a loss
//
// It takes strike price and premium in $$$ per share and returns break-even point
// regarding to the option type (opt)
pub fn break_even_point(opt: OptionKind, strike: f64, premium: Option<f64>) -> MathResult {
    if strike < 0.0 {
        return Err(MathError::NonPositiveStrike);
    }
    let premium = premium.unwrap_or_default();
    if premium < 0.0 {
        return Err(MathError::NonPositivePremium);
    }
    match opt {
        OptionKind::Call => Ok(strike + premium),
        OptionKind::Put => Ok(strike - premium),
    }
}

// payoff calculates current market price of an option ($$$ per share)
//
// It takes position (pos) strike price (str), stock (st) and premium (pr) in $$$ per share and returns profit/lose
// from buying options regarding to the option  type (opt)
pub fn payoff(
    pos: Position,
    opt: OptionKind,
    strike: f64,
    stock: f64,
    premium: Option<f64>,
) -> MathResult {
    if strike < 0.0 {
        return Err(MathError::NonPositiveStrike);
    }
    if stock < 0.0 {
        return Err(MathError::NonPositiveStock);
    }
    let premium = premium.unwrap_or_default();
    if premium < 0.0 {
        return Err(MathError::NonPositivePremium);
    }

    match pos {
        Position::Long => match opt {
            OptionKind::Call => Ok((stock - strike).max(0.0)),
            OptionKind::Put => Ok((strike - stock).max(0.0)),
        },
        Position::Short => match opt {
            OptionKind::Call => Ok(premium - (stock - strike).max(0.0)),
            OptionKind::Put => Ok(premium - (strike - stock).max(0.0)),
        },
    }
}

fn norm_dist(z: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let t2 = t.powi(2);
    let y = t
        * (0.319381530 - 0.356563782 * t + (1.781477937 - 1.821255978 * t + 1.330274429 * t2) * t2);

    if z > 0.0 {
        return 1.0 - (-((2.0 * PI).ln() + z.powi(2)) * 0.5).exp() * y;
    }
    return (-((2.0 * PI).ln() + -z.powi(2)) * 0.5).exp() * y;
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn err_with_negative_strike() {
        let errored = break_even_point(OptionKind::Call, -50.0, Some(10.0)).is_err();
        assert!(errored);
    }

    #[test]
    fn err_with_negative_premium() {
        let errored = break_even_point(OptionKind::Call, 50.0, Some(-10.0)).is_err();
        assert!(errored);
    }

    #[test]
    fn call_with_premium() {
        let result = break_even_point(OptionKind::Call, 50.0, Some(10.0)).unwrap();
        assert_eq!(result, 60.0);
    }

    #[test]
    fn call_without_premium() {
        let result = break_even_point(OptionKind::Call, 50.0, None).unwrap();
        assert_eq!(result, 50.0);
    }

    #[test]
    fn put_with_premium() {
        let result = break_even_point(OptionKind::Put, 50.0, Some(10.0)).unwrap();
        assert_eq!(result, 40.0);
    }

    #[test]
    fn put_without_premium() {
        let result = break_even_point(OptionKind::Put, 50.0, None).unwrap();
        assert_eq!(result, 50.0);
    }

    #[test]
    fn positive_norm_dist() {
        let result = norm_dist(0.39);
        assert_eq!(result, 0.6517316779654632);
    }

    #[test]
    fn zero_norm_dist() {
        let result = norm_dist(0.0);
        assert_eq!(result, 0.49999999947519136);
    }

    #[test]
    fn negative_norm_dist() {
        let result = norm_dist(-0.39);
        assert_eq!(result, 0.4054806781620218);
    }

    #[test]
    fn call_price() {
        let bsm =
            BlackScholesModel::new(OptionKind::Call, 58.0, 60.0, 0.035, 0.2, 0.5, Some(0.0125));
        let result = bsm.price().unwrap();

        assert_eq!(result, 4.556957304081674);
    }
    #[test]
    fn put_price() {
        let bsm =
            BlackScholesModel::new(OptionKind::Put, 58.0, 60.0, 0.035, 0.2, 0.5, Some(0.0125));
        let result = bsm.price().unwrap();

        assert_eq!(result, 1.758568520665552);
    }
}
