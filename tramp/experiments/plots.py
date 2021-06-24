import itertools
import numpy as np
import matplotlib.pyplot as plt
from ..utils.misc import array2complex


def plot_function(f, xlim=(-3, 3)):
    x = np.linspace(xlim[0], xlim[1], 100)
    plt.plot(x, f(x))
    plt.title(f"{f.__name__}")


def plot_compare(x_true, x_pred):
    fig, ax = plt.subplots(1, 1, figsize=(4,4))
    ax.plot(np.arange(100), x_true[:100], label="x true")
    ax.plot(np.arange(100), x_pred[:100], label="x pred")
    ax.legend()
    fig.tight_layout()


def plot_compare_complex(x_true, x_pred):
    if x_true.dtype != "complex":
        x_true = array2complex(x_true)
    if x_pred.dtype != "complex":
        x_pred = array2complex(x_pred)
    fig, axs = plt.subplots(1, 2, sharey=False, sharex=False)
    axs[0].plot(np.angle(x_true), np.angle(x_pred), '.')
    axs[0].set(xlabel="phase x true", ylabel="phase x pred")
    axs[1].plot(np.absolute(x_true), np.absolute(x_pred), '.')
    axs[1].set(xlabel="modulus x true", ylabel="modulus x pred")
    fig.tight_layout()


def format_kwargs(sep, pattern, **kwargs):
    out = sep.join([
        pattern % (key, val) for key, val in kwargs.items()
    ])
    return out


def as_query(query_format, **kwargs):
    formats = {
        "pandas": (" & ", "%s=='%s'"),
        "vega": (" & ", "(datum.%s=='%s')"),
        "title": (" ", "%s=%s")
    }
    if query_format in formats:
        sep, pattern = formats[query_format]
    else:
        raise ValueError("query_format should be one of %s" % formats.keys())
    query = format_kwargs(sep, pattern, **kwargs)
    return query


AES_PALETTE = {
    "linestyle": ['-', '--', '-.', ':'],
    "marker": ['.', 'x', '+', 'o', 'v', '^', '<', '>', 's', 'D'],
    "color": [
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
        (1.0, 0.4980392156862745, 0.054901960784313725),
        (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
        (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
        (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
        (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
        (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
        (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
        (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
        (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)
    ]
}


def get_plot_instructions(data, aes_fields):
    def get_values(field):
        return sorted(data[field].unique())
    field_choices = {
        field: get_values(field)
        for field in aes_fields.values() if field
    }
    field_records = [
        {key: value for key, value in zip(field_choices.keys(), record_values)}
        for record_values in itertools.product(*field_choices.values())
    ]
    instructions = []
    for field_record in field_records:
        query = as_query("pandas", **field_record)
        pos = dict(row=0, column=0)
        options = {}
        title = label = ""
        for aes in ["row", "column", "color", "marker", "linestyle"]:
            field = aes_fields[aes]
            if field:
                value = field_record[field]
                field_values = get_values(field)
                idx_value = field_values.index(value)
                if aes in ["row", "column"]:
                    title += f"{field}={value} "
                    pos[aes] = idx_value
                if aes in ["color", "marker", "linestyle"]:
                    label += f"{field}={value} "
                    palette = AES_PALETTE[aes]
                    options[aes] = palette[idx_value]
        instructions.append(dict(
            query=query, row=pos["row"], column=pos["column"],
            options=options, title=title, label=label
        ))
    return instructions


def replace(label, mapping):
    "Replace key-> val in string `label`, for all key->val in `mapping`"
    if mapping:
        for old, new in mapping.items():
            label = label.replace(old, new)
    return label


def qplot(data, x, y,
          color=None, column=None, row=None, marker=None, linestyle=None,
          xlog=False, ylog=False, xlim=None, ylim=None,
          y_markers=None, sharex=True, sharey=True, figsize=4,
          y_legend=False,
          rename=None, font_size=12, usetex=False
          ):
    # check args
    y_multiple = isinstance(y, list)
    if y_multiple:
        if not isinstance(y_markers, list) or len(y) != len(y_markers):
            raise ValueError("y_markers must be a list of same length as y")
        if marker is not None:
            raise ValueError("cannot use marker in y_multiple mode")
        if linestyle is not None:
            raise ValueError("cannot use linestyle in y_multiple mode")
    plt.rc('text', usetex=usetex)
    plt.rc('font', family='serif', size=font_size)
    # get plotting instructions
    aes_fields = dict(
        color=color, column=column, row=row, marker=marker, linestyle=linestyle
    )
    instructions = get_plot_instructions(data, aes_fields)
    # set figure
    nrows = max(instruction["row"] for instruction in instructions) + 1
    ncols = max(instruction["column"] for instruction in instructions) + 1
    if isinstance(figsize, float) or isinstance(figsize, int):
        figsize = (figsize * ncols, figsize * nrows)
    fig, axs = plt.subplots(
        nrows, ncols, squeeze=False, figsize=figsize,
        sharex=sharex, sharey=sharey
    )
    # set scales
    if xlog:
        plt.xscale("log")
    if ylog:
        plt.yscale("log")
    # iterate over plotting instructions
    for instruction in instructions:
        # get data
        if instruction["query"]:
            df = data.query(instruction["query"])
        else:
            df = data
        # get ax
        ax = axs[instruction["row"], instruction["column"]]
        # plot ax
        if y_multiple:
            # iterate over y
            for i, (y_var, y_marker) in enumerate(zip(y, y_markers)):
                # set label
                if y_legend:
                    label = instruction["label"] + " " + y_var
                else:
                    label = instruction["label"] if i == 0 else ""
                label = replace(label, rename)
                # set aes
                if y_marker in AES_PALETTE["linestyle"]:
                    instruction["options"].update(linestyle=y_marker, marker="")
                elif y_marker in AES_PALETTE["marker"]:
                    instruction["options"].update(linestyle="", marker=y_marker)
                else:
                    raise ValueError(f"unkown marker {y_marker}")
                # plot
                ax.plot(
                    df[x], df[y_var], **instruction["options"], label=label
                )
        else:
            # set label
            if y_legend:
                label = instruction["label"] + " " + y
            else:
                label = instruction["label"]
            label = replace(label, rename)
            # set aes
            if not linestyle and not marker:
                instruction["options"].update(linestyle="-", marker="")
            if linestyle and not marker:
                instruction["options"].update(marker="")
            if not linestyle and marker:
                instruction["options"].update(linestyle="")
            # plot
            ax.plot(
                df[x], df[y], **instruction['options'], label=label
            )
        # set title
        title = replace(instruction['title'], rename)
        ax.set(title=title)
    # set limits
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    # set xlabel and ylabel
    if y_legend:
        ylabel = ""
    else:
        ylabel = ", ".join(y) if y_multiple else y
    xlabel = replace(x, rename)
    ylabel = replace(ylabel, rename)
    any_label = any(instruction['label'] for instruction in instructions)
    for ax in axs.ravel():
        ax.set(xlabel=xlabel, ylabel=ylabel)
        if any_label or y_legend:
            ax.legend()
    fig.tight_layout()
