###### Ковертер лежит в папке model
###### Все библиотеки и их зависимости занимают достаточно много памяти ~1.3Гб. Поэтому рекомедую сосздать виртуальное окружение
    
    Команды для создания виртуального окружения и обнвления pip
    python -m venv .env
    python -m pip install --upgrade pip

    Запуск виртуальной среды
    .\.env\Scripts\Activate.ps1 или .\.env\Scripts\activate.bat


    Команды установки необходимых библиотек
    pip install ultralytics
    pip install --upgrade executorch

###### Запуск конвертации модели
	python .\pt2pte.py <путь до файла модели>.pt 