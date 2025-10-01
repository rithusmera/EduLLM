import json

def sectioning(file_path):

    with open(file_path,'r', encoding='utf-8') as file:
        content = file.read()

    raw_sections = content.split('###')

    sections = []

    for i in raw_sections:
        if i == "":
            continue

        lists = i.split('\n',1)
        
        if len(lists)==2:
            title = lists[0].strip()
            body = lists[1].strip()
        else:
            title = lists[0].strip()
            body = ''

        sections.append({'title':title, 'body':body})

    return sections

def chunking(sections, max_length= 300, overlap= 50):

    chunked_sections = []
    section_id_counter = 1

    for section in sections:

        title = section.get('title','')
        body = section.get('body','')

        words = body.split(' ')
        total_words = len(words)

        if total_words < max_length:
            chunked_sections.append(
                {
                    'Title': title,
                    'Chunk No.': 1,
                    'Total Chunks': 1,
                    'Section ID': f"sec{section_id_counter}",
                    'Body': body
                }
            )
        
        else:

            chunk_list = []
            start = 0
            while True:
                end = min(start + max_length, total_words)
                chunk_words = words[start:end]
                chunk_body = ' '.join(chunk_words)
                chunk_list.append(chunk_body)
                if end == total_words:
                    break
                start += max_length- overlap

            for i, chunk in enumerate(chunk_list):
                chunked_sections.append({
                        'Title': title,
                        'Chunk No.': i+1,
                        'Total Chunks': len(chunk_list),
                        'Section ID': f'sec{section_id_counter}',
                        'Body': chunk}
                )

        section_id_counter+= 1
    
    return chunked_sections

file_path = r'.\Text Files\Sample_text.txt'
sections = sectioning(file_path)
chunked_sections = chunking(sections, 300, 50)

with open(r'.\Chunks\Sample_chunks.json', 'w', encoding='utf-8') as file:
    json.dump(chunked_sections, file, ensure_ascii=False, indent=4)