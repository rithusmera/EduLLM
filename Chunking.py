import json
CLASS = '11'
SUBJECT = 'Physics'
CHAPTER = 'Motion in a Straight Line'
MAX_LENGTH = 300
OVERLAP = 50

def sectioning(file_path):

    with open(file_path,'r', encoding='utf-8') as file:
        content = file.read()

    raw_sections = content.split('## ')
    sections = []

    for i in raw_sections:
        if i == "":
            continue

        lists = i.split('\n\n',1)
        
        if len(lists)==2:
            title = lists[0].strip()
            body = lists[1].strip()
        else:
            title = ''
            body = lists[0].strip()

        sections.append({'title':title, 'body':body})

    return sections

def chunk_subitems(text, max_length=MAX_LENGTH, overlap=OVERLAP):
    words = text.split()
    total_words = len(words)
    chunks = []

    if total_words <= max_length:
        chunks.append(text)
    else:
        start = 0
        while True:
            end = min(start + max_length, total_words)
            chunks.append(' '.join(words[start:end]))
            if end == total_words:
                break
            start += max_length - overlap
    return chunks

def chunking(sections):

    chunked_sections = []
    section_id_counter = 1

    for section in sections:

        sec_title = section.get('title','')
        sec_body = section.get('body','')

        sub_items = sec_body.split('###')
        sub_counter = 1

        for sub in sub_items:
            if not sub.strip():
                continue

            lines = sub.split('\n\n', 1)

            if len(lines) == 2:
                sub_title = lines[0].strip()
                sub_body = lines[1].strip()
            else:
                sub_title = ''
                sub_body = lines[0].strip()
        
            lower_title = sub_title.lower()
            if lower_title.startswith('example'):
                ctype = 'example'
            elif lower_title.startswith('table'):
                ctype = 'table'
            elif lower_title.startswith('figure'):
                ctype = 'figure'
            elif lower_title.startswith('exercise'):
                ctype = 'exercise'
            else:
                ctype = 'text'

            chunks = chunk_subitems(sub_body)

            for idx, chunk_text in enumerate(chunks):
                chunk_id = f"{ctype}_{section_id_counter}_{sub_counter}_{idx+1}"
                chunked_sections.append({
                    'id': chunk_id,
                    'type': ctype,
                    'class': CLASS,
                    'subject': SUBJECT,
                    'chapter': CHAPTER,
                    'section_title': sec_title,
                    'section_id': f"sec{section_id_counter}",
                    'parent_section_id': f"sec{section_id_counter}" if ctype in ('figure','table') else None,
                    'title': sub_title,
                    'content': chunk_text,
                    'chunk_no': idx+1,
                    'total_chunks': len(chunks)
                })
            sub_counter += 1
        section_id_counter += 1
    return chunked_sections

file_path = r'Text Files\Physics\Class 11\Chapter3.txt'
sections = sectioning(file_path)
chunked_sections = chunking(sections)

with open(r'Chapters\Physics\Class 11\Chap2_Chunked.json', 'w', encoding='utf-8') as file:
    json.dump(chunked_sections, file, ensure_ascii=False, indent=4)