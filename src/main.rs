use ::rand::prelude::*;
use macroquad::prelude::*;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

pub mod colormap;
use crate::colormap::*;


fn colormap(value: f64, colormap: &[[u8; 4]]) -> [u8; 4] {
    let idx = (127. * value + 127.) as usize;
    colormap[idx]
}

fn sink_edges(value: f64, xnorm: f64, ynorm: f64) -> f64 {
    let x = (xnorm - 0.5).abs();
    let y = (ynorm - 0.5).abs();
    value - 5.*(x*x+y*y)
}

// Return cardinal neighbors of point, [West, North, East, South]
fn get_neighbors(point: (usize, usize), xmax: usize, ymax: usize) -> Vec<Option<(usize, usize)>> {
    let mut output = vec![];
    if point.0 > 0 {
        output.push(Some((point.0 - 1, point.1)));
    } else {
        output.push(None);
    }
    if point.1 > 0 {
        output.push(Some((point.0, point.1 - 1)));
    } else {
        output.push(None);
    }
    if point.0 < xmax - 1 {
        output.push(Some((point.0 + 1, point.1)));
    } else {
        output.push(None);
    }
    if point.1 < ymax - 1 {
        output.push(Some((point.0, point.1 + 1)));
    } else {
        output.push(None);
    }
    output
}

// Define positive x gradient as East, positive y gradient as South.
fn map_gradient(map: &[Vec<f64>]) -> Vec<Vec<(f64, f64)>> {
    let mut output = vec![];
    for y in 0..map.len() {
        let mut row = vec![(0.0, 0.0); map[y].len()];
        for x in 0..map[y].len() {
            let h = map[y][x];
            let n = get_neighbors((x, y), map[y].len(), map.len());
            let mut dx = 0.0;
            if let Some(nn) = n[0] {
                dx += (h - map[nn.1][nn.0]) / 2.0;
            }
            if let Some(nn) = n[2] {
                dx += (map[nn.1][nn.0] - h) / 2.0;
            }
            let mut dy = 0.0;
            if let Some(nn) = n[1] {
                dy += (h - map[nn.1][nn.0]) / 2.0;
            }
            if let Some(nn) = n[3] {
                dy += (map[nn.1][nn.0] - h) / 2.0;
            }
            row[x] = (dx, dy);
        }
        output.push(row);
    }
    output
}

fn talus_erosion_iteration(map: &mut [Vec<f64>], limit: f64, amount: f64) {
    // Compute heightmap gradients
    let grads = map_gradient(map);

    for y in 1..map.len()-1 {
        for x in 1..map[y].len()-1 {
            let (dx, dy) = grads[y][x];
            if dx.abs() > dy.abs() && dx.abs() > limit {
                // Erode in dx direction
                let xother = if dx > 0.0 {
                    x - 1
                } else {
                    x + 1
                };
                map[y][x] -= amount;
                map[y][xother] += amount;
            } else if dy.abs() > limit {
                // Erode in dy direction
                let yother = if dy > 0.0 {
                    y - 1
                } else {
                    y + 1
                };
                map[y][x] -= amount;
                map[yother][x] += amount;
            }
        }
    }
}

fn colorize(map: &[Vec<f64>]) -> Vec<u8> {
    let mut rgba = vec![];
    for row in 0..map.len() {
        for col in 0..map[row].len() {
            let value = colormap(map[row][col], &greyscale);
            rgba.push(value[0]);
            rgba.push(value[1]);
            rgba.push(value[2]);
            rgba.push(value[3]);
        }
    }
    rgba
}

fn hydraulic_erosion_iteration(map: &mut [Vec<f64>], steps: usize, limit: f64, amount: f64) {
    // Spawn random drop location according to rainfall distribution
    let grads = map_gradient(map);
    let mut x = ::rand::random_range(1..=map[0].len()-1);
    let mut y = ::rand::random_range(1..=map.len()-1);

    // move drop some number of steps (while height > waterlevel), or some limit for evaporation
    if map[y][x] < 0.0 {
        return;
    }
    let mut sediment = 0.0;
    for i in 0..steps {
        if x == 0 || y == 0 || x == map[0].len() - 1 || y == map.len() - 1 {
            break;
        }
        let (dx, dy) = grads[y][x];
        if dx.abs() > dy.abs() {
            if dx.abs() > limit {
                // "Fast" so pick up sediments, move in X direction
                // Erode in dx direction
                map[y][x] -= amount;
                sediment += amount;
            } else {
                // "Slow" so drop sediment
                if sediment > 0.0 {
                    sediment -= amount;
                    map[y][x] += amount;
                }
            }
            x = if dx > 0.0 {
                x - 1
            } else {
                x + 1
            };
        } else if dy.abs() > limit {
            if dy.abs() > limit {
                // "Fast" so pick up sediments, move in Y direction
                // Erode in dy direction
                map[y][x] -= amount;
                sediment += amount;
            } else {
                // "Slow" so drop sediment
                if sediment > 0.0 {
                    sediment -= amount;
                    map[y][x] += amount;
                }
            }
            y = if dy > 0.0 {
                y - 1
            } else {
                y + 1
            };
        }
    }

    map[y][x] += sediment;
}

#[macroquad::main("RPCG")]
async fn main() {
    let fbm = Fbm::<Perlin>::new(1337)
        .set_octaves(5)
        .set_frequency(0.005);
    let w = screen_width();
    let h = screen_height();
    
    // Base heightmap
    let mut heightmap = vec![vec![0.0; w as usize]; h as usize];
    for row in 0..h as usize {
        for col in 0..w as usize {
            let value = fbm.get([col as f64, row as f64]);
            let value2 = sink_edges(value, col as f64 / w as f64, row as f64 / h as f64);
            heightmap[row][col] = value2;
        }
    }
    let before = heightmap.clone();

    // Erosion
    if true {
        let limit = 1e-2;
        let amount = 1e-3;
        for _ in 0..100 {
            talus_erosion_iteration(&mut heightmap, limit, amount);
        }
    }
    if false {
        // Very WIP, not great.
        let drop_steps = 1000;
        let limit = 1e-2;
        let amount = 3e-2;
        for _ in 0..200 {
            hydraulic_erosion_iteration(&mut heightmap, drop_steps, limit, amount);
        }
    }

    // Colorize
    let before_rgba = colorize(&before);
    let rgba = colorize(&heightmap);

    // Textures
    let before_texture = Texture2D::from_rgba8(w as u16, h as u16, &before_rgba);
    let texture = Texture2D::from_rgba8(w as u16, h as u16, &rgba);

    let mut screen = 0;

    loop {
        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        if is_key_pressed(KeyCode::Space) {
            screen += 1;
            screen %= 2;
        }

        // Draw
        clear_background(BLACK);
        if screen == 0 {
            draw_texture(&texture, 0., 0., WHITE);
            draw_text("After", 20., 20., 20., YELLOW);
        } else if screen == 1 {
            draw_texture(&before_texture, 0., 0., WHITE);
            draw_text("Before", 20., 20., 20., YELLOW);
        }
        next_frame().await
    }
}
