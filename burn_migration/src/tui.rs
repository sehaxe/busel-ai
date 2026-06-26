// Burn's TUI renderer — live progress bar with loss, tok/s, ETA.
use std::{cell::RefCell, io::IsTerminal, sync::Arc};

use burn::{
    data::dataloader::Progress,
    train::{
        metric::{MetricAttributes, MetricDefinition, MetricEntry, MetricId, NumericAttributes, NumericEntry, SerializedEntry},
        renderer::{MetricState, MetricsRenderer, MetricsRendererTraining, ProgressType, TrainingProgress, tui::TuiMetricsRendererWrapper},
    },
};

enum Inner {
    Active(RefCell<TuiMetricsRendererWrapper>),
    Noop,
}

pub struct TuiHandle {
    inner: Inner,
}

impl TuiHandle {
    pub fn new(_max_steps: usize) -> Self {
        if !std::io::stdout().is_terminal() {
            eprintln!("[busel-burn] tui-full requires a terminal — running without TUI");
            return Self { inner: Inner::Noop };
        }

        let interrupter = burn::train::Interrupter::new();
        let mut renderer = TuiMetricsRendererWrapper::new(interrupter, None);

        renderer.register_metric(MetricDefinition {
            metric_id: MetricId::new(Arc::new("loss".into())),
            name: "Loss".into(),
            description: None,
            attributes: MetricAttributes::Numeric(NumericAttributes {
                unit: None,
                higher_is_better: false,
            }),
        });

        Self { inner: Inner::Active(RefCell::new(renderer)) }
    }

    pub fn tick(&self, step: usize, max: usize, loss: f32, tok_s: f64) {
        let Inner::Active(r) = &self.inner else { return };
        let mut r = r.borrow_mut();
        let metric_id = MetricId::new(Arc::new("loss".into()));
        let entry = MetricEntry::new(metric_id, SerializedEntry::new(
            format!("{:.4}", loss), format!("{:.4}", loss),
        ));
        r.update_train(MetricState::Numeric(entry, NumericEntry::Value(loss as f64)));

        r.render_train(
            TrainingProgress {
                progress: Some(Progress { items_processed: step, items_total: max }),
                global_progress: Progress { items_processed: step, items_total: max },
                iteration: Some(step),
            },
            vec![ProgressType::Value { tag: "tok/s".into(), value: tok_s as usize }],
        );
    }

    pub fn done(&self) {
        let Inner::Active(r) = &self.inner else { return };
        let mut r = r.borrow_mut();
        let _ = r.on_train_end(None);
    }
}
