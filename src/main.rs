use ::rand::prelude::*;
use macroquad::prelude::*;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

pub mod colormap;
use crate::colormap::*;

#[derive(Clone, Debug)]
struct Map<T> {
    dims: (usize, usize),
    data: Vec<T>,
}

impl<T: Clone + Default> Map<T> {
    fn zeroed(height: usize, width: usize) -> Self {
        Self {
            dims: (height, width),
            data: vec![T::default(); height * width],
        }
    }

    fn get(&self, x: usize, y: usize) -> &T {
        &self.data[x*self.dims.1 + y]
    }
}

fn colormap(value: f64, colormap: &[[u8; 4]]) -> [u8; 4] {
    let idx = (255. * value) as usize;
    colormap[idx]
}

// value is heightmap on [0.0, 1.0]
// xnorm is x position normalized to [0.0, 1.0]
// ynorm is y position normalized to [0.0, 1.0]
fn sink_edges(value: f64, xnorm: f64, ynorm: f64) -> f64 {
    let x = 2. * (xnorm - 0.5);
    let y = 2. * (ynorm - 0.5);
    let output = value * (1.0 - (x * x + y * y));
    if output < 0.0 { 0.0 } else { output }
}

// Return cardinal neighbors of point, [North, East, South, West]
fn get_neighbors(point: (usize, usize), xmax: usize, ymax: usize) -> Vec<Option<(usize, usize)>> {
    let mut output = vec![];
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
    if point.0 > 0 {
        output.push(Some((point.0 - 1, point.1)));
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

    for y in 1..map.len() - 1 {
        for x in 1..map[y].len() - 1 {
            let (dx, dy) = grads[y][x];
            if dx.abs() > dy.abs() && dx.abs() > limit {
                // Erode in dx direction
                let xother = if dx > 0.0 { x - 1 } else { x + 1 };
                map[y][x] -= amount;
                map[y][xother] += amount;
            } else if dy.abs() > limit {
                // Erode in dy direction
                let yother = if dy > 0.0 { y - 1 } else { y + 1 };
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
    let mut x = ::rand::random_range(1..=map[0].len() - 1);
    let mut y = ::rand::random_range(1..=map.len() - 1);

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
            x = if dx > 0.0 { x - 1 } else { x + 1 };
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
            y = if dy > 0.0 { y - 1 } else { y + 1 };
        }
    }

    map[y][x] += sediment;
}

fn generate_heightmap() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let seed: u32 = ::rand::random();
    println!("seed: {}", seed);
    let fbm = Fbm::<Perlin>::new(seed).set_octaves(5).set_frequency(0.005);
    let w = screen_width();
    let h = screen_height();

    // Base heightmap
    let mut heightmap = vec![vec![0.0; w as usize]; h as usize];
    for row in 0..h as usize {
        for col in 0..w as usize {
            // value on [0.0, 1.0]
            let value = 0.5 * fbm.get([col as f64, row as f64]) + 0.5;
            let value2 = sink_edges(
                value,
                col as f64 / (w - 1.) as f64,
                row as f64 / (h - 1.) as f64,
            );
            if value2 < 0.0 || value2 > 1.0 {
                eprintln!("value2: {}", value2);
                panic!("value2 not in [0.0, 1.0]");
            }
            heightmap[row][col] = value2;
        }
    }
    let before = heightmap.clone();

    // Erosion
    if true {
        let limit = 1e-3;
        let amount = 1e-3;
        for _ in 0..10 {
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
    (heightmap, before)
}

fn heightmap_to_texture(map: &[Vec<f64>]) -> Texture2D {
    let w = screen_width();
    let h = screen_height();

    // Colorize
    let rgba = colorize(&map);

    // Textures
    Texture2D::from_rgba8(w as u16, h as u16, &rgba)
}

fn normal(pos: (usize, usize), map: &Map<f64>) -> (f64, f64) {
    let h = map.get(pos.0, pos.1);
    let dims = map.dims;
    let n = get_neighbors(pos, dims.0, dims.1);
    let mut dy = 0.0;
    if let Some(nn) = n[0] {
        // North
        let other_h = map.get(nn.0, nn.1);
        dy += (other_h - h) / 2.0;
    }
    if let Some(nn) = n[2] {
        // South
        let other_h = map.get(nn.0, nn.1);
        dy += (h - other_h) / 2.0;
    }
    let mut dx = 0.0;
    if let Some(nn) = n[1] {
        // East
        let other_h = map.get(nn.0, nn.1);
        dx += (other_h - h) / 2.0;
    }
    if let Some(nn) = n[3] {
        // West
        let other_h = map.get(nn.0, nn.1);
        dx += (h - other_h) / 2.0;
    }
    (dx, dy)
}

#[derive(Clone, Copy, Debug)]
struct WaterUnit {
    volume: f64,
    sediment: f64,
    density: f64,
    pos: (usize, usize),
    velocity: (f64, f64),
}

impl WaterUnit {
    fn new(x: usize, y:usize) -> Self {
        Self {
            volume: 1.0,
            sediment: 0.0,
            density: 1.0,
            pos: (x, y),
            velocity: (0.0, 0.0),
        }
    }

    fn descend(&mut self, map: &mut Map<f64>, track: &mut Map<bool>) {
        let dt = 0.1;
        let friction = 0.1;
        let h0 = map.get(self.pos.0, self.pos.1);
        let n = normal(self.pos, map);
        let mass = self.volume * self.density;
        self.velocity.0 += dt * n.0 / mass;
        self.velocity.1 += dt * n.1 / mass;
        self.pos.0 = (self.pos.0 as f64  + dt * self.velocity.0) as usize;
        self.pos.1 = (self.pos.1 as f64  + dt * self.velocity.1) as usize;
        self.velocity.0 *= 1.0 - dt * friction;
        self.velocity.1 *= 1.0 - dt * friction;
        let h1 = map.get(self.pos.0, self.pos.1);
        let speed: f64 = (self.velocity.0.powi(2) + self.velocity.1.powi(2)).sqrt();
        let mut ceq = self.volume * speed * (h0 - h1);
        if ceq < 0.0 {
            ceq = 0.0;
        }
        let cdiff = ceq - 
    }

    fn flood(&mut self, map: &mut [Vec<f64>]) {

    }
}

fn hydraulic_erosion_iteration2(map: &mut [Vec<f64>], stream: &mut [Vec<f64>], pool: &mut [Vec<f64>], track: &mut [Vec<bool>], steps: usize) {
    for i in 0..steps {
        let mut x = ::rand::random_range(1..=map[0].len() - 1);
        let mut y = ::rand::random_range(1..=map.len() - 1);
        let mut drop = WaterUnit::new(x, y);
    }
}

#[macroquad::main("RPCG")]
async fn main() {
    let mut w = screen_width();
    let mut h = screen_height();
    request_new_screen_size(1920., 1080.);
    let (mut heightmap, mut before) = generate_heightmap();

    // Track tells where water ran this iteration
    let mut track = Map::<bool>::zeroed(w as usize, h as usize);

    // Pool tells where pools are in the grid, and how deep they are.
    let mut pool = Map::<f64>::zeroed(w as usize, h as usize);

    // Stream tells where water is moving in the grid, and how deep they are.
    let mut stream = Map::<f64>::zeroed(w as usize, h as usize);

    // Textures
    let mut before_texture = heightmap_to_texture(&before);
    let mut texture = heightmap_to_texture(&heightmap);

    let mut screen = 0;

    loop {
        if screen_width() != w || screen_height() != h {
            w = screen_width();
            h = screen_height();
            (heightmap, before) = generate_heightmap();
            before_texture = heightmap_to_texture(&before);
            texture = heightmap_to_texture(&heightmap);
        }
        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        if is_key_pressed(KeyCode::Space) {
            screen += 1;
            screen %= 2;
        }

        if is_key_pressed(KeyCode::Enter) {
            println!("Generating new map...");
            (heightmap, before) = generate_heightmap();
            before_texture = heightmap_to_texture(&before);
            texture = heightmap_to_texture(&heightmap);
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
