topics = [['mulher',
  'homem',
  'menino',
  'escola',
  'amigo',
  'tempo',
  'amigar',
  'epoca',
  'sala',
  'namorar'],[
 'pessoa',
  'coisa',
  'vidar',
  'amigo',
  'tempo',
  'casar',
  'gente',
  'vezar',
  'problema',
  'entao'],[
 'vidar',
  'coisa',
  'casar',
  'pessoa',
  'problema',
  'tempo',
  'familia',
  'trabalhar',
  'empregar',
  'dinheiro'],[
 'cursar',
  'escola',
  'aula',
  'partir',
  'faculdade',
  'provar',
  'professorar',
  'ensinar',
  'noto',
  'sociedade'],[
 'pessoa',
  'suicidio',
  'ansiedade',
  'formar',
  'tratamento',
  'remedios',
  'psiquiatro',
  'pensamento',
  'morte',
  'terapia']]

labels = [",".join(topic[:3]) for topic in topics]

import pandas as pd
import matplotlib.pyplot as plt

x = [0.15, 0.3, 0.45, 0.6, 0.75]
x2 = [0.95, 0.3, 0.6, 0.8, 0.75]
y = [2.56422, 3.77284, 3.52623, 3.51468, 3.02199]

fig, ax = plt.subplots(1)
ax.scatter(x, y, c=['red', 'blue', 'green', 'purple', 'orange'])
ax.scatter(x2, y, c=['red', 'blue', 'green', 'purple', 'orange'])

for i, txt in enumerate(labels):
    ax.annotate(txt, (x[i], y[i]))
plt.axis('off')
plt.show()