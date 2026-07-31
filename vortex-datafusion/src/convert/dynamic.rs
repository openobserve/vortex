// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_schema::Schema;
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
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;
use vortex::scalar_fn::fns::operators::CompareOperator;

use crate::convert::FromDataFusion;

/// Returns the single non-null child used by a Top-K dynamic filter.
pub(super) fn topk_dynamic_child(
    expr: &PhysicalExprRef,
    schema: &Schema,
) -> Option<PhysicalExprRef> {
    let dynamic = expr.downcast_ref::<DynamicFilterPhysicalExpr>()?;
    let children = dynamic.children();
    let [child] = children.as_slice() else {
        return None;
    };

    if child.nullable(schema).ok()? {
        return None;
    }

    Some(Arc::clone(child))
}

/// Converts a DataFusion single-column Top-K filter into two conservative Vortex dynamic bounds.
///
/// DataFusion initially represents the filter as `true`, so the sort direction is not available
/// until the first threshold arrives. Each bound therefore defaults to `true` and only activates
/// when the current expression exactly matches its comparison operator.
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

    Ok(and(
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
            rhs_dtype,
            true,
            lhs,
        ),
    ))
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
        let comparison = current.downcast_ref::<df_expr::BinaryExpr>()?;

        if *comparison.op() != expected_operator || !comparison.left().eq(&self.child) {
            return None;
        }

        let literal = comparison.right().downcast_ref::<df_expr::Literal>()?;
        let scalar = Scalar::from_df(literal.value());
        if !scalar.dtype().eq_ignore_nullability(&self.rhs_dtype) {
            return None;
        }

        scalar.into_value()
    }
}
