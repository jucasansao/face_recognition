import click
import itertools
import multiprocessing
import os
import re
import sys

import numpy as np
import PIL.Image

import face_recognition.api as face_recognition


def scan_known_people(known_people_folder):
    known_names = []
    known_face_encodings = []

    for file in image_files_in_folder(known_people_folder):
        basename = os.path.splitext(os.path.basename(file))[0]
        img = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 1:
            click.echo(f"WARNING: More than one face found in {file}. Only considering the first face.")

        if len(encodings) == 0:
            click.echo(f"WARNING: No faces found in {file}. Ignoring file.")
        else:
            known_names.append(basename)
            known_face_encodings.append(encodings[0])

    return known_names, known_face_encodings


def print_result(filename, name, distance, show_distance=False):
    if show_distance:
        print(f"{filename},{name},{distance}")
    else:
        print(f"{filename},{name}")


def test_image(image_to_check, known_names, known_face_encodings, tolerance=0.6, show_distance=False):
    unknown_image = face_recognition.load_image_file(image_to_check)

    # Scale down image if it's giant so things run a little faster
    if max(unknown_image.shape) > 1600:
        pil_img = PIL.Image.fromarray(unknown_image)
        pil_img.thumbnail((1600, 1600), PIL.Image.LANCZOS)
        unknown_image = np.array(pil_img)

    unknown_encodings = face_recognition.face_encodings(unknown_image)

    for unknown_encoding in unknown_encodings:
        distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)
        result = list(distances <= tolerance)

        if True in result:
            for is_match, name, distance in zip(result, known_names, distances):
                if is_match:
                    print_result(image_to_check, name, distance, show_distance)
        else:
            print_result(image_to_check, "unknown_person", None, show_distance)

    if not unknown_encodings:
        print_result(image_to_check, "no_persons_found", None, show_distance)


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


def process_images_in_process_pool(images_to_check, known_names, known_face_encodings, number_of_cpus, tolerance, show_distance):
    processes = None if number_of_cpus == -1 else number_of_cpus

    # macOS will crash due to a bug in libdispatch if you don't use 'forkserver'
    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")

    pool = context.Pool(processes=processes)

    function_parameters = zip(
        images_to_check,
        itertools.repeat(known_names),
        itertools.repeat(known_face_encodings),
        itertools.repeat(tolerance),
        itertools.repeat(show_distance),
    )

    pool.starmap(test_image, function_parameters)


@click.command()
@click.argument("known_people_folder")
@click.argument("image_to_check")
@click.option("--cpus", default=1, help='Number of CPU cores to use in parallel. -1 means "use all in system".')
@click.option("--tolerance", default=0.6, help="Tolerance for face comparisons. Default is 0.6. Lower this if you get multiple matches for the same person.")
@click.option("--show-distance", default=False, type=bool, help="Output face distance. Useful for tweaking tolerance setting.")
def main(known_people_folder, image_to_check, cpus, tolerance, show_distance):
    known_names, known_face_encodings = scan_known_people(known_people_folder)

    if os.path.isdir(image_to_check):
        if cpus == 1:
            for image_file in image_files_in_folder(image_to_check):
                test_image(image_file, known_names, known_face_encodings, tolerance, show_distance)
        else:
            process_images_in_process_pool(image_files_in_folder(image_to_check), known_names, known_face_encodings, cpus, tolerance, show_distance)
    else:
        test_image(image_to_check, known_names, known_face_encodings, tolerance, show_distance)


if __name__ == "__main__":
    main()
