# vladyslav_gadzikhanov

1)  
    1.0) Не совсем понимаю, что нужно было сделать в этом пункте, надеюсь, что прописал необходимые вещи тут.  
    1.1) Самооценка ниже.  
    1.2) Разведочный анализ данных:  
&emsp	     1.2.1)  mlops_hw1/notebooks/manual_eda: ручками сделанный.  
&emsp	     1.2.2)  mlops_hw1/notebooks/script_eda: с помощью скрипта.  
    1.3) Функция для тренировки модели: models/model_fit_predict.py/train_model, инструкция ниже.                  			                       1.4) Функция для прогнозов: models/model_fit_predict.py/predict_model, инструкция ниже.  
    1.5) Модульная структура.  
    1.6) Логгеры:  
	     1.6.1) Задается конфигурация: mlops_hw1/models/defining_logger_configuration.py.  
	     1.6.2) Файл для записи: mlops_hw1/file_handler.log.  
    1.7) Тесты: mlops_hw1/tests.  
    1.8) Генерация приближенных к реальным данных для тестов: mlops_hw1/data_generation.  
    1.9) Конфиги: mlops_hw1/configs.  
	     1.9.1) first_config.yaml: CatBoostClassifier.  
	     1.9.2) second_config.yaml: LogisticRegression.  
    1.10) Датаклассы: mlops_hw1/enties.  
    1.11) Самописный трансформер:  
	     1.11.1) Сам трансформер: mlops_hw1/custom_transformer/transformer.py.  
	     1.11.2) Тесты: mlops_hw1/tests/test_transformer.  
    1.12) Зависимости: mlops_hw1/requirements.txt.  
    1.13) ---.  
  
Сделано все, кроме 13 пункта и доп. части.  
  
2) п. 3, 4 =>  запуск из mlops_hw1/..  
	 python3 -m mlops_hw1.models.model_fit_predict config_path flag  
  
	 1. config_path = mlops_hw1/configs/first_config.yaml  
	 2. flag = (fit / predict)  
