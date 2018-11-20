right=0
test_output = list()
test_input = list()
correct_answer = list()

test_set = open('../resource/test-set.txt','r', encoding="gbk").read().splitlines()

for i in range(0,len(test_set)):
	if (i % 2) == 0:
	    test_input.append(test_set[i])
	else:
		correct_answer.append(test_set[i])

with open("../data/input-evaluation.txt","w") as f:
	f.write('')
with open("../data/correct-answer.txt","w") as f:
	f.write('')

for i in range(0,len(test_input)):
    with open("../data/input-evaluation.txt","a") as f:
        f.write(test_input[i])
        f.write('\n')
    with open("../data/correct-answer.txt","a") as f:
        f.write(correct_answer[i])
        f.write('\n')

output_evaluation = open('../data/output-evaluation.txt','r', encoding="UTF-8").read()
correct_answer = open('../data/correct-answer.txt','r', encoding="UTF-8").read()
output_evaluation = output_evaluation.replace(' ','').replace('\n','')
correct_answer = correct_answer.replace(' ','').replace('\n','')

for i in range(0, len(output_evaluation)):
    if correct_answer[i] == output_evaluation[i]:
        right += 1

print('字符数 = %d' % (i+1))
print('正确数 = %d' % right)
print('正确率 = %.1f%%' %(100*right/(i+1)))