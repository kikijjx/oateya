import re

with open('C:\\Users\\nikita\\Desktop\\vscode\\lab1\\vesna.txt', 'r', encoding='cp1251') as file:
    text = file.read()

text = text.replace('\n', '')
text = text.replace('!', '.')
text = text.replace('?', '.')
text = text.lower()
text = re.sub(r'(?<!\w)-|-(?!\w)|[^\w\s.-]', '', text)
text = text.split(sep='.')

words = []
for i in text:
    i = i.strip()
    if i:
        words.append(i.split())
print(words)

print('всего слов:')
print(len(sum(words, [])))
print('уникальных слов:')
print(len(set(sum(words, []))))
print('предложений:')
print(len(words))
print(max(sum(words, []), key=len))

srs = ['а','и', 'в', 'на', 'с', 'по', 'к', 'у', 'за', 'из', 'от', 'до', 'под', 'над', 'о', 'об', 'обо', 'для', 'без', 'при', 'про', 'через', 'между', 'во', 'со', 'под', 'над', 'обо', 'для', 'без', 'при', 'про', 'через', 'между', 'во', 'со']
print(min(filter(lambda s: s.lower() not in srs, sum(words, [])), key=len))


lengths = [len(word) for word in sum(words, [])]
sredlength = sum(lengths) / len(lengths)
sredlength_sort = sorted(lengths)
median = sredlength_sort[len(sredlength_sort) // 2]

pred_lengths = [len(sentence) for sentence in words]
pred_sredlength = sum(pred_lengths) / len(pred_lengths)
pred_sredlength_sort = sorted(pred_lengths)
pred_median = pred_sredlength_sort[len(pred_sredlength_sort) // 2]
print('средняя длина слов:')
print(sredlength)
print('медианная длина слов:')
print(median)
print('средняя длина предложений:')
print(pred_sredlength)
print('медианная длина предложений:')
print(pred_median)