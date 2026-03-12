import codecs

def main():
    try:
        lines = codecs.open('test_all.log', 'r', 'utf-16le').read().splitlines()
    except Exception as e:
        print("Failed to read utf-16:", e)
        return

    results = []
    for line in lines:
        if any(x in line for x in ['Testing ', 'Prompt:', 'Status:', 'Exception', 'Error:']):
            # Filter out random logger internal traces that might have 'Exception'
            if '|' not in line:
                results.append(line)
            elif 'Status:' in line or 'Testing ' in line or 'Prompt:' in line:
                results.append(line)
                
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
