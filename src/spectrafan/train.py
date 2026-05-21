"""Training loop.

Default config from the paper: RMSprop, lr 1e-5, decay 0.99, weight decay 1e-8,
momentum 0.999, 200 epochs, batch 4, loss = 0.5 * CE + 0.5 * Dice.
Metrics: IoU, Dice, pixel accuracy. Training cost: wall-clock per epoch and
time-to-target-IoU, not just epoch count.
"""
