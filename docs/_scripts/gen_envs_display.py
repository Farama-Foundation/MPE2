import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GIF_REL_PATH = "../_static/img/videos"
GIF_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "_static", "img", "videos"))

all_envs = {
    "mpe2": [
        "simple",
        "simple_adversary",
        "simple_crypto",
        "simple_formation",
        "simple_line",
        "simple_push",
        "simple_reference",
        "simple_speaker_listener",
        "simple_spread",
        "simple_tag",
        "simple_world_comm",
        "collect_treasure",
    ]
}


def create_grid_cell(type_id, env_id, base_path):
    # The relative image path assumes that the list will be at /environment/env_type/
    gif_file = f"{type_id}_{env_id}.gif"
    abs_path = os.path.join(GIF_DIR, gif_file)
    gif_tag = (
        f'<img src="{GIF_REL_PATH}/{gif_file}">'
        if os.path.exists(abs_path)
        else ""
    )
    return f"""
            <a href="/environments/{env_id}">
                <div class="env-grid__cell">
                    <div class="cell__image-container">
                        {gif_tag}
                    </div>
                    <div class="cell__title">
                        <span>{' '.join(env_id.split('_')).title()}</span>
                    </div>
                </div>
            </a>
    """


def generate_page(env_type_id, env_list, limit=-1, base_path=""):
    cells = [create_grid_cell(env_type_id, env_id, base_path) for env_id in env_list]
    non_limited_page = limit == -1 or limit >= len(cells)
    if non_limited_page:
        cells = "\n".join(cells)
    else:
        cells = "\n".join(cells[:limit])

    more_btn = (
        '<a href="./complete_list"><button class="more-btn">See More Environments</button></a>'
        if not non_limited_page
        else ""
    )
    return f"""
<!DOCTYPE html>
<html>
    <body>
        <div class="env-grid">
            {cells}
        </div>
        {more_btn}
    </body>
</html>
    """


if __name__ == "__main__":
    """
    python gen_envs_display
    """

    if len(sys.argv) > 1:
        type_arg = sys.argv[1]

    for key in all_envs.keys():
        env_list = all_envs[key]
        envs_path = os.path.join(os.path.dirname(__file__), "..", "environments")
        page = generate_page(key, env_list)
        fp = open(os.path.join(envs_path, "list.html"), "w+", encoding="utf-8")
        fp.write(page)
        fp.close()
