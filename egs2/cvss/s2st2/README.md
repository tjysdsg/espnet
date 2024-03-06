# Docs

Useful link: https://github.com/tjysdsg/espnet/pull/21/files view the diff between u2u branch and master to help
understand.

## s2st2/s2st.sh

s2st2 recipe is based on s2st made by Jiatong. The main difference is that it supports discrete unit input.
Otherwise, the recipe is almost the same as the original one, except for example, some input layers in the model.

Follow input option `use_discrete_input` and `use_discrete_unit` to read the code.

`use_discrete_unit` was the original option in s2st1 recipe created by Jiatong, indicating if the s2st model outputs
discrete units or spectrogram.

I added `use_discrete_input` to determine if the input is units.

Checkout `espnet2/tasks/s2st.py` to understand how this option modifies the input loading process.
Also checkout `espnet2/s2st/espnet_model.py` and search for `u2u` and `discrete_input`.

Pay attention to how the discrete unit dictionary is used for both the model input and the output during data loading.
I did this because as I recall the paper uses a unified set of units.

## Other

Other than `s2st2`, I'd read `espnet2/s2st/espnet_model.py` carefully and compare the implementation with
with https://arxiv.org/abs/2107.05604

If you want you can also read the UnitY paper.

## Work that needs to be done

The u2u paper uses voxpopuli as its data.
Future work AFAIK that needs to be done to at least reproduce the u2u paper:

- Adapt existing voxpopuli data loading scripts to support loading multiple languages at once.
  The key challenge is that not only we have two audio sample for a single training sample, we also have to remember
  their language ids.
  Might take some time to think of a good way to represent this in Kaldi format.
- Dataloader has to be modified accordingly. Mainly the `MutliTokenizerCommonPreprocessor` class that is used
  in `s2st.py`.
- Combine other speech tasks (see paper Figure 2) into the existing training loop. This could be the most challenging
  part. I hadn't started to think about it.