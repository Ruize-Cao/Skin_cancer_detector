import tensorflow as tf


"""Custom focal loss used to reduce the effect of class imbalance."""


class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    """Sparse categorical focal loss for integer class labels."""

    def __init__(
        self,
        gamma=2.0,
        alpha=0.5,
        from_logits=True,
        name="sparse_categorical_focal_loss",
        reduction="sum_over_batch_size",
    ):
        super().__init__(name=name, reduction=reduction)

        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        """Compute focal loss from true labels and predicted logits/probabilities."""

        y_true = tf.cast(y_true, tf.int32)

        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        y_true_one_hot = tf.one_hot(
            y_true,
            depth=tf.shape(y_pred)[-1],
        )

        epsilon = 1e-7
        y_pred = tf.clip_by_value(
            y_pred,
            epsilon,
            1.0 - epsilon,
        )

        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)

        focal_weight = self.alpha * tf.pow(
            1 - y_pred,
            self.gamma,
        )

        loss = focal_weight * cross_entropy

        return tf.reduce_mean(
            tf.reduce_sum(loss, axis=-1)
        )

    def get_config(self):
        """Make the custom loss serializable in saved Keras models."""
        config = super().get_config()
        config.update(
            {
                "gamma": self.gamma,
                "alpha": self.alpha,
                "from_logits": self.from_logits,
            }
        )
        return config
