import re
import sys
import pysam


def parse_cs_tag(tag, refseq, readseq, debug=False):
    ret = ""
    match = 0
    mismatch = 0
    deletions = 0
    insertions = 0
    refpos = 0
    readpos = 0
    refstr = ""
    readstr = ""
    p = re.findall(r":([0-9]+)|(\*[a-z][a-z])|(=[A-Za-z]+)|(\+[A-Za-z]+)|(\-[A-Za-z]+)", tag)
    for i, x in enumerate(p):
        #print(x)
        if len(x[0]):
            q = int(x[0])
            if debug: print("match:", i, q)
            match+= int(q)
            refstr+=refseq[refpos:refpos+q]
            refpos+=q
            readstr+=readseq[readpos:readpos+q]
            readpos+=q
            ret+= "|" * q
        if len(x[1]):
            q = int((len(x[1])-1)/2)
            if debug: print("mismatch:", i, q)
            mismatch+= q
            refstr+=refseq[refpos:refpos+q]
            refpos+=q
            readstr+=readseq[readpos:readpos+q]
            readpos+=q
            ret+= "*" * q
        if len(x[2]):
            if debug: print("FAIL")
        if len(x[3]):
            q = len(x[3])-1
            if debug: print("insertion:", i, q, x[3])
            insertions+=q
            refstr+="-"*q
            readstr+=readseq[readpos:readpos+q]
            readpos+=q
            ret+= " " * q
        if len(x[4]):
            q = len(x[4])-1
            if debug: print("deletion:", i, q)
            deletions+=q
            refstr+=refseq[refpos:refpos+q]
            refpos+=q
            readstr+="-"*q
            ret+= " " * q

    return match, insertions, deletions, mismatch


def main():
    # SAM file to parse
    # sam_file = sys.argv[1]
    sam_file = "/home/alex/Documents/tmp/ngram-1-aln.sam"
    ref_file = "/home/alex/OneDrive/phd-project/basecaller-data/dRNA/1_DivideTrainTestFast5s/0_0_RawData/gencode.v34.transcripts.fa"
    out_file = sam_file.replace(".sam", ".tsv")

    # Record stats
    stats = []

    # Load reference file
    ref_file = pysam.FastaFile(ref_file)

    # Parse SAM file
    sam_file = pysam.AlignmentFile(sam_file, "r")
    with open(out_file, "w") as out:
        out.write("read_id\tref_name\tn_match\tn_ins\tn_del\tn_sub\n")
        
        n_unmapped = 0
        for read in sam_file:
            if read.is_unmapped:
                n_unmapped += 1
                continue

            # Get reference sequence
            ref_seq = ref_file.fetch(read.reference_name, read.reference_start, read.reference_end)
            
            if not read.seq:
                print("TODO: NO QUERY SEQ")

            # Extract relevant info from alignment record
            read_id = read.qname
            ref_name = read.reference_name.split("|")
            transcript = ref_name[0]
            seq = read.seq[read.query_alignment_start:read.query_alignment_end]
            
            # RODAN version
            cs_tag = read.get_tag("cs")
            n_match, n_ins, n_del, n_sub = parse_cs_tag(cs_tag, ref_seq, seq)
        
            # My version
            my_match = 0
            my_ins = 0
            my_del = 0
            my_sub = 0
            for char in read.cigar:
                op = char[0]
                count = char[1]
                if op == 0:
                    my_match += count
                elif op == 1:
                    my_ins += count
                elif op == 2:
                    my_del += count
            
            nm = read.get_tag("NM")
            my_sub = nm - my_ins - my_del
            my_match -= my_sub

            print(f"Match: {my_match}\t{n_match}")
            print(f"Ins: {my_ins}\t{n_ins}")
            print(f"Del: {my_del}\t{n_del}")
            print(f"Sub: {my_sub}\t{n_sub}")

            

            # stats.append([n_match, n_ins, n_del, n_sub])
            # out.write(f"{read_id}\t{transcript}\t{n_match}\t{n_ins}\t{n_del}\t{n_sub}")
    
    for read in stats:
        print(read)


if __name__ == "__main__":
    main()
