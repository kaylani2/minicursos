# minicurso_ml_sbrc2020

CICIDS: http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/.

NSL-KDD: http://205.174.165.80/CICDataset/NSL-KDD/Dataset/


```bash
foo@bar:~$ git clone https://github.com/kaylani2/minicurso_ml_sbrc2020
foo@bar:~$ wget 205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip
foo@bar:~$ wget 205.174.165.80/CICDataset/NSL-KDD/Dataset/NSL-KDD.zip
foo@bar:~$ pip3 install -r requirements.txt
```

No dataset NSL-KDD, para que a biblioteca Pandas importe corretamente os nomes dos atributos de cada amostra, adicionar a linha seguinte ao início dos arquivos .txt:

duration,protocol_type,service,flag,src_bytes,dst_bytes,land,wrong_fragment,urgent,hot,num_failed_logins,logged_in,num_compromised,root_shell,su_attempted,num_root,num_file_creations,num_shells,num_access_files,num_outbound_cmds,is_host_login,is_guest_login,count,srv_count,serror_rate,srv_serror_rate,rerror_rate,srv_rerror_rate,same_srv_rate,diff_srv_rate,srv_diff_host_rate,dst_host_count,dst_host_srv_count,dst_host_same_srv_rate,dst_host_diff_srv_rate,dst_host_same_src_port_rate,dst_host_srv_diff_host_rate,dst_host_serror_rate,dst_host_srv_serror_rate,dst_host_rerror_rate,dst_host_srv_rerror_rate,class,severity
