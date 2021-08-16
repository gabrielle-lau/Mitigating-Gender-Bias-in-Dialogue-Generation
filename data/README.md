# Data
## Hostile Sexism Data
Twitter Hostile Sexism tweet IDs are provided by [Jha et al.](https://github.com/AkshitaJha/NLP_CSS_2017)
## StereoSet Data
StereoSet data are provided by [Nadeem et al.](https://github.com/moinnadeem/StereoSet/tree/master/data)
## Blender Data
ConvAI2, Empathetic Dialogues, Wizard of Wikipedia and Blended Skill Talk are downloaded through ParlAI when ``display_data`` is called for the first time.
Sample command:
```
parlai display_data --task convai2
```
The [data with gender bais tokens](https://github.com/facebookresearch/ParlAI/tree/47765c3256a3e91ba4ae50fdc42250ad5f0c0ecf/parlai/tasks/genderation_bias) can be obtained by
```
parlai display_data --task genderation_bias:controllable_task:convai2
```
