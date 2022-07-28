proc=$(ps -ef | grep "[p]ython3 auto_tag_api.py")
pid=$(echo ${proc} | cut -d " " -f2)
echo pid : ${pid}
`kill -9 ${pid}`