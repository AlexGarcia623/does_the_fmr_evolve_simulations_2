import warnings
warnings.filterwarnings("ignore")

file_names = [
    'Figure2.py'   , 'Figure3.py'   , 'Figure4.py'   ,
    'Figure5.py'   , 'Figure6.py'   , 'Figure8.py'   ,
    'AppendixA1.py', 'AppendixB1.py', 'AppendixB2.py',
    'AppendixC1.py', 'AppendixC2.py', 'AppendixD1.py'
]

for file_name in file_names:
    with open(file_name, 'r') as file:
        print('\n')
        to_output = f'#### Starting: {file_name} ####'
        print('#'*len(to_output))
        print(to_output)
        print('#'*len(to_output))
        print('\n')
        exec(file.read())
        print('\n')
        print('!'*len(to_output))
        print(f'!!!! Finished: {file_name} !!!!')
        print('!'*len(to_output))