"""Render zero-based throughput overview charts without plotting dependencies."""

from html import escape


def throughput_svg(field, hosts, rows):
    """Return an accessible paired-host bar chart from unrounded aggregates."""
    values = [value for _, pair in rows for value in pair if value is not None]
    maximum = max(values) if values else 1.0
    width, left, plot = 960, 260, 580
    height = 135 + len(rows) * 58
    colors = ("#2563eb", "#b45309")
    title = f"{field}: packed operations at 256 KiB"
    description = ("Throughput in GiB/s; higher is better. Both hosts share a linear "
                   "scale starting at zero. Exact measurements appear in the tables below.")
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           'role="img" aria-labelledby="title description">',
           f'<title id="title">{escape(title)}</title>',
           f'<desc id="description">{escape(description)}</desc>',
           f'<rect width="{width}" height="{height}" fill="#fff"/>',
           '<g font-family="system-ui, sans-serif" fill="#172033">',
           f'<text x="20" y="28" font-size="20" font-weight="600">{escape(title)}</text>']
    for index, host in enumerate(hosts):
        x = 20 + index * 460
        out += [f'<rect x="{x}" y="45" width="12" height="12" fill="{colors[index]}"/>',
                f'<text x="{x + 20}" y="56" font-size="14">{escape(host)}</text>']
    for step in range(5):
        value = maximum * step / 4
        x = left + plot * step / 4
        out += [f'<path d="M{x:.2f} 85 V{height - 30}" stroke="#d9dee7"/>',
                f'<text x="{x:.2f}" y="78" text-anchor="middle" font-size="12">'
                f'{value:.1f}</text>']
    for index, (operation, pair) in enumerate(rows):
        y = 100 + index * 58
        out.append(f'<text x="{left - 12}" y="{y + 17}" text-anchor="end" '
                   f'font-size="14">{escape(operation)}</text>')
        for host_index, value in enumerate(pair):
            bar_y = y + host_index * 20
            if value is None:
                out.append(f'<text x="{left + 5}" y="{bar_y + 13}" font-size="12">Not measured</text>')
                continue
            bar_width = plot * value / maximum
            out += [f'<rect x="{left}" y="{bar_y}" width="{bar_width:.2f}" '
                    f'height="16" fill="{colors[host_index]}">'
                    f'<title>{escape(hosts[host_index])}: {value:.3f} GiB/s</title></rect>',
                    f'<text x="{left + bar_width + 6:.2f}" y="{bar_y + 13}" '
                    f'font-size="12">{value:.3g}</text>']
    out += [f'<text x="{left}" y="{height - 8}" font-size="12">GiB/s · higher is better · zero-based shared scale</text>',
            '</g></svg>']
    return "\n".join(out) + "\n"
