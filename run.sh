for layer1 in 10 20 30
	for layer2 in 5 10 15
		for lr in 1e-5 1e-4 1e-3
			echo "${layer1} ${layer2} ${lr}"
			python main4.py --filename "news_categorization_reu.csv" --model_name "NN" --nn_layer1 ${layer1} --nn_layer2 ${layer2} --nn_lr ${lr}
		do
	do
do