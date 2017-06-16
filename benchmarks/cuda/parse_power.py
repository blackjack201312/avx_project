pw_file = open('power.log')
content = pw_file.readlines()
dst_file = open('parse_power.txt', 'w')

gtx980 = [content[s + 1].strip() for s, value in enumerate(content) if "GTX 980" in value]
gtx1080 = [content[s + 1].strip() for s, value in enumerate(content) if "GTX 1080" in value]

records = [gtx980[s] + gtx1080[s] + '\n' for s in range(len(gtx980))]
dst_file.writelines(records)

pw_file.close()
dst_file.close()


