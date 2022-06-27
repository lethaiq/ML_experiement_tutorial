for seed in 12 17
do
	for layer1 in 10 30
	do
		for layer2 in 5 15
		do
			for lr in 1e-5 1e-4 1e-3
			do
				echo "${layer1} ${layer2} ${lr}"
				python main4.py --filename "news_categorization_reu.csv" --model_name "NN" --nn_layer1 ${layer1} --nn_layer2 ${layer2} --nn_lr ${lr}
			done
		done
	done
done