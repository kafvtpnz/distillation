
������ ������������� ����������� �����������

� �������� ������ ������������ ����������� Bert ��������� �� ������� ������ ������������� ������������ � ������ ���� � ������
������� Russian Language Toxic Comments Dataset https://www.kaggle.com/blackmoon/russian-language-toxic-comments

����� ������� ������ ����� ������� (https://drive.google.com/file/d/1e4ZLyUGU22RwjSDHnTKHBe0XwsPr6xl1/view?usp=sharing)
� �������� ���� ����� Bert-� � ������� model_bert



�������� CNN �� ���������� Bert 

distillation.py

���������:
--alpha, type=float, default=0.5  �����������, �������� ����������� loss �� ������ � loss �� Bert-� ��� �������� CNN
--batch_size, type=int, default=4 ������ �����
--epochs, type=int, default=10,   ���� ��������
--lr, type=float, default=0.002,  ��������� �������� ��������
--max_len, type=int, default=50,  ����� ������������������, ������� ��������� � CNN


������������ Bert
bert_pred.py --text "��� �����"

������������ CNN (��������� ���� � �������� model_cnn ����� ��������� �����������)
cnn_pred.py --text "��� �����"


� �������� 10 ���� �������� ������ ��������� �� ���������
AccuracyCNN: 0.739612188365651, AccuracyBert: 0.8961218836565097