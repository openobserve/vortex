// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use datafusion_common::Result as DFResult;
use datafusion_common::exec_datafusion_err;
use datafusion_expr::Operator as DFOperator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr::expressions as df_expr;
use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
use vortex::dtype::DType;
use vortex::expr::Expression;
use vortex::expr::and;
use vortex::expr::dynamic;
use vortex::expr::is_null;
use vortex::expr::or;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;
use vortex::scalar_fn::fns::operators::CompareOperator;

use crate::convert::FromDataFusion;

/// Returns the single child used by a Top-K dynamic filter.
///
/// DataFusion's Parquet reader evaluates nullable dynamic predicates directly. Vortex accepts
/// single-column Top-K filters as best-effort predicates and conservatively retains nulls; the
/// exact DataFusion predicate above the scan remains responsible for null ordering semantics.
pub(super) fn topk_dynamic_child(expr: &PhysicalExprRef) -> Option<PhysicalExprRef> {
    let dynamic = expr.downcast_ref::<DynamicFilterPhysicalExpr>()?;
    let children = dynamic.children();
    let [child] = children.as_slice() else {
        return None;
    };

    Some(Arc::clone(child))
}

/// Converts a DataFusion single-column Top-K filter into two conservative Vortex dynamic bounds.
///
/// DataFusion initially represents the filter as `true`, so the sort direction is not available
/// until the first threshold arrives. Each bound therefore defaults to `true` and only activates
/// when the current expression matches either `child <op> literal` or DataFusion's nullable
/// `is_null(child) OR child <op> literal` form.
pub(super) fn convert_topk_dynamic_filter(
    expr: &PhysicalExprRef,
    child: PhysicalExprRef,
    lhs: Expression,
    rhs_dtype: DType,
) -> DFResult<Expression> {
    if expr.downcast_ref::<DynamicFilterPhysicalExpr>().is_none() {
        return Err(exec_datafusion_err!(
            "Expected a DataFusion dynamic filter expression"
        ));
    }

    let state = Arc::new(TopKDynamicState {
        filter: Arc::clone(expr),
        child,
        rhs_dtype: rhs_dtype.clone(),
    });
    let lt_state = Arc::clone(&state);
    let gt_state = Arc::clone(&state);

    let bounds = and(
        dynamic(
            CompareOperator::Lt,
            move || lt_state.threshold(DFOperator::Lt),
            rhs_dtype.clone(),
            true,
            lhs.clone(),
        ),
        dynamic(
            CompareOperator::Gt,
            move || gt_state.threshold(DFOperator::Gt),
            rhs_dtype.clone(),
            true,
            lhs.clone(),
        ),
    );

    // A nullable Top-K predicate may switch between a plain comparison and DataFusion's
    // `is_null(child) OR comparison` shape as its threshold changes. The Vortex dynamic scalar
    // only carries the comparison value, so always retaining nulls is the conservative choice.
    // DataFusion re-evaluates the exact dynamic predicate above this best-effort scan filter.
    Ok(if rhs_dtype.is_nullable() {
        or(is_null(lhs), bounds)
    } else {
        bounds
    })
}

struct TopKDynamicState {
    filter: PhysicalExprRef,
    child: PhysicalExprRef,
    rhs_dtype: DType,
}

impl TopKDynamicState {
    fn threshold(&self, expected_operator: DFOperator) -> Option<ScalarValue> {
        let dynamic = self.filter.downcast_ref::<DynamicFilterPhysicalExpr>()?;
        let current = dynamic.current().ok()?;
        let comparison = self.comparison(&current, expected_operator)?;

        let literal = comparison.right().downcast_ref::<df_expr::Literal>()?;
        let scalar = Scalar::from_df(literal.value());
        if !scalar.dtype().eq_ignore_nullability(&self.rhs_dtype) {
            return None;
        }

        scalar.into_value()
    }

    fn comparison<'a>(
        &self,
        current: &'a PhysicalExprRef,
        expected_operator: DFOperator,
    ) -> Option<&'a df_expr::BinaryExpr> {
        if let Some(comparison) = self.direct_comparison(current, expected_operator) {
            return Some(comparison);
        }

        let compound = current.downcast_ref::<df_expr::BinaryExpr>()?;
        if *compound.op() != DFOperator::Or {
            return None;
        }

        self.null_aware_comparison(compound.left(), compound.right(), expected_operator)
            .or_else(|| {
                self.null_aware_comparison(compound.right(), compound.left(), expected_operator)
            })
    }

    fn null_aware_comparison<'a>(
        &self,
        null_expr: &PhysicalExprRef,
        comparison: &'a PhysicalExprRef,
        expected_operator: DFOperator,
    ) -> Option<&'a df_expr::BinaryExpr> {
        let is_null = null_expr.downcast_ref::<df_expr::IsNullExpr>()?;
        if !is_null.arg().eq(&self.child) {
            return None;
        }

        self.direct_comparison(comparison, expected_operator)
    }

    fn direct_comparison<'a>(
        &self,
        expr: &'a PhysicalExprRef,
        expected_operator: DFOperator,
    ) -> Option<&'a df_expr::BinaryExpr> {
        let comparison = expr.downcast_ref::<df_expr::BinaryExpr>()?;
        (*comparison.op() == expected_operator && comparison.left().eq(&self.child))
            .then_some(comparison)
    }
}
