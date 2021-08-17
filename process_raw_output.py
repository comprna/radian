import os

def main():

    raw_dir = "./to_process"

    for raw_file in os.listdir(raw_dir):
        out_fn = f"{raw_file.split('.')[0]}-out.txt"

        with open(f"{raw_dir}/{raw_file}", "r") as raw:

            with open(out_fn, "w") as out:
                lines = raw.readlines()

                # Exclude gadi output at start and end of file
                lines = lines[32:-13]

                # # Exclude extra print statements on odd lines
                # lines = lines[::2]

                for line in lines:
                    out.write(line)


if __name__ == "__main__":
    main()
