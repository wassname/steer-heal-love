# 2026-06-04
  
  # steer-heal-love
  
  Hypothesis: you can distill a steering vector into LoRA weights and "heal" the incoherency the vector injects by regularising the training (KL to base, or weight decay). Then loop and see what multiple rounds give you.
  
  The crux: KL-to-base penalises all drift, persona shift included. The bet is that incoherency drift is large and erratic while the persona shift is small and systematic, so KL kills the incoherency preferentially. If that's wrong, we just trade persona strength for coherence instead of getting both.
  
  ## Source
  
  Found this interesting: https://r.jina.ai/https://arxiv.org/html/2606.00995v1
  
  They use steering vectors as an internal perturbation to generate synthetic data, which is what weight steering does too. But:
  
  - they use single completions, not pairs
  - they don't measure incoherency (they should)
  - they only use one direction: base to pos, not neg to pos
  
  So this is similar to weight steering, except you heal with KL or WD instead of taking the direction between two adapters.
  
  ## Method
  
  1. Pick a positive persona, e.g. `pos = "you do not defer to authority and instead stick to principle no matter your involvement"`.
  2. Build the steering vector from the distance `hs_base -> hs_pos` (hidden states). This is normal mean-mass contrastive steering
  3. Generate completions with this vector.
    - Drop completions that are incoherent, or that verbalise the trait instead of enacting it (we want the model to act it out, not narrate "I am someone who..."). Filter as much as we can.
    - We might be able to dial the vector down for long trajectories. Could we even backtrack an incoherent vector and replay parts with less intervention? Or just cosine-gate at test time.
  4. Train a LoRA on these completions, could be just 50 completions and 2 epochs. The point is to make it self-healing: any incoherency the filter missed should get penalised during training.
    - Regularise with KL or WD so the outputs, distribution, or weights don't shift too far from base. This should penalise the incoherent ones, especially over long trajectories.
  5. Bake in the LoRA adapter. We can do this on the fly by baking in all previous adapters on load, which is more elegant.
  6. Eval the checkpoint on https://github.com/wassname/tinymfv.
  7. If it works, loop. We could even do this online, GRPO-style per batch, or iteratively. Iterative is simpler to start.
  
  ## Eval
  
  Plot the tinymfv progress over time on the auth vs care axis, with a subplot for a coherence measure. tinymfv gives a few: `p_ans_any` (best), `json_is_valid`, `ppx_json`.

# 2026

❯ /arj interesting that is potentially healed... although did it heal or just undo? the radio of
dAuth vs coherence migth be the wa yto measure that

weird that authority went backawards and that's it's a small effect overall? since coherence
hardly went down (I would expect down to 0.95).

so my new goal for you make the steering strong enougth to hard a bigger effect even if it's
0.95 coherence
--
if we need to switch to -Care or +Tradition then we can if the model response better
oh also mean mass shift kind of sucks? at least with small amount of thinking, so you might want
to make tinymfv use 128 or 256 think tokens of the cdefualt 64 is unreliable. shouldn't be
needed but plan C record it in spec

oh perhaps there's a better steering, cosine gates or something, or the SVD (look at results in
steering-lite and consider that some are much harder to bake, e.g. how to base cosine gating...
you can't) so this is plan D record in spec

---

❯ so the first gate should now be
  does steering actually change the target on the eval whilebeing coherent? if not you need to
  iterate and think and fix

  2nd gate can we filter? check qualitative samples on the borderline on the filter

  3rd gate does the lora learn differen't and coherent examples

  so have you actually got a steering vector the works

❯ look if filter + lora works that great, we can ablate later. but the real uncertainty is
  getting it working!
  we might have too strong regularisaiton on the lora, what would you expect to see then?
  what is the steering vector is to weak or to strong what would you expect to see there? what
  if it was to imprecises and just a bad pareto trade, waht would you observe?
  you should think about what you would observe at each gate in the likely possible outcomes
  including subtle ones. then tell me if you measure it and if you see it
  this includes eval results but also qualitative judgmeent from you
