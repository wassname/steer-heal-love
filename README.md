# steer-heal-love

What if you can **steer**, **heal** the steering and repeat untill alignment (**love**). 
<!--(Staring Julia Roberts: If your wife has made you watch eat, pray love too many times, you will understand the reference)-->

Hypothesis: you can distill a steering vector into LoRA weights and "heal" the incoherency the vector injects by regularising the training (KL to base, or weight decay). Then loop and see what multiple rounds give you.

The crux: KL-to-base penalises all drift, persona shift included. The bet is that incoherency drift is large and erratic while the persona shift is small and systematic, so KL kills the incoherency preferentially. If that's wrong, we just trade persona strength for coherence instead of getting both.


## Experiment

1. Pick a positive persona, e.g. `pos = "you do not defer to authority and instead stick to principle no matter your involvement"`.
2. Build the steering vector from the distance `hs_base -> hs_pos` (hidden states). This is normal mean-mass contrastive steering
3. Generate completions with this vector.
   - Drop completions that are incoherent, or that verbalise the trait instead of enacting it (we want the model to act it out, not narrate "I am someone who..."). Filter as much as we can.
   - **Q0 can we filter?**
   - We might be able to dial the vector down for long trajectories. Could we even backtrack an incoherent vector and replay parts with less intervention? Or just cosine-gate at test time.
4. Train a LoRA on these completions, could be just 50 completions and 2 epochs. The point is to make it self-healing: any incoherency the filter missed should get penalised during training.
   - Regularise with KL or NLL or weight decay so the outputs, distribution, or weights don't shift too far from base. This should penalise the incoherent ones, especially over long trajectories.
   - **Q1: can we heal incoherency?**
5. Bake in the LoRA adapter. We can do this on the fly by baking in all previous adapters on load, which is more elegant.
6. Eval the checkpoint on https://github.com/wassname/tinymfv.
7. If it works, loop. We could even do this online, GRPO-style per batch, or iteratively. Iterative is simpler to start.
- **Q2: is it coherent over a loop?**
- **Q3: does it keep moving consistency in a direction?**


Most likely failure modes: 
- It fails at the 4 Q's above
- doesn't beat a prompting baseline

## Motovation:

If it works it will be a novel alignment method that works without label and might be resistant to deceptive alignment

## Eval

Plot the tinymfv progress over time on the auth vs care axis


# Results

TODO insert plot
