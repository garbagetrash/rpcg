#![allow(unused, non_upper_case_globals)]

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
    fn zeroed(width: usize, height: usize) -> Self {
        Self {
            dims: (width, height),
            data: vec![T::default(); height * width],
        }
    }

    fn get(&self, x: usize, y: usize) -> &T {
        &self.data[y*self.dims.0 + x]
    }

    fn get_mut(&mut self, x: usize, y: usize) -> &mut T {
        &mut self.data[y*self.dims.0 + x]
    }
}

fn colormap(value: f64, colormap: &[[u8; 4]]) -> [u8; 4] {
    let idx = (255. * value.clamp(0.0, 1.0)) as usize;
    colormap[idx]
}

// value is heightmap on [0.0, 1.0]
// xnorm is x position normalized to [0.0, 1.0]
// ynorm is y position normalized to [0.0, 1.0]
fn sink_edges(value: f64, xnorm: f64, ynorm: f64) -> f64 {
    let x = 2. * (xnorm - 0.5);
    let y = 2. * (ynorm - 0.5);
    let output = value * (1.0 - (x * x + y * y));
    output.clamp(0.0, 1.0)
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

fn colorize(map: &Map<f64>) -> Vec<u8> {
    let mut rgba = vec![];
    for &_value in &map.data {
        let value = colormap(_value, &greyscale);
        rgba.push(value[0]);
        rgba.push(value[1]);
        rgba.push(value[2]);
        rgba.push(value[3]);
    }
    rgba
}

fn generate_heightmap() -> Map<f64> {
    let seed: u32 = ::rand::random();
    println!("seed: {}", seed);
    let fbm = Fbm::<Perlin>::new(seed).set_octaves(5).set_frequency(0.005);
    let w = screen_width();
    let h = screen_height();

    // Base heightmap
    let mut heightmap = Map::<f64>::zeroed(w as usize, h as usize);
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
            *heightmap.get_mut(col, row) = value2;
        }
    }
    heightmap
}

fn heightmap_to_texture(map: &Map<f64>) -> Texture2D {
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

        // Hardcoded trash
        let dt = 0.1;
        let friction = 0.1;
        let deposition_rate = 0.1;
        let evaporation_rate = 0.1;

        // Update physical parameters
        let n = normal(self.pos, map);
        let mass = self.volume * self.density;
        let h0 = map.get(self.pos.0, self.pos.1);
        self.velocity.0 += dt * n.0 / mass;
        self.velocity.1 += dt * n.1 / mass;
        self.pos.0 = (self.pos.0 as f64 + dt * self.velocity.0) as usize;
        self.pos.1 = (self.pos.1 as f64 + dt * self.velocity.1) as usize;
        self.velocity.0 *= 1.0 - dt * friction;
        self.velocity.1 *= 1.0 - dt * friction;
        let speed: f64 = (self.velocity.0.powi(2) + self.velocity.1.powi(2)).sqrt();
        let cdiff = {
            let h1 = map.get(self.pos.0, self.pos.1);
            let mut ceq: f64 = self.volume * speed * (h0 - h1);
            if ceq < 0.0 {
                ceq = 0.0;
            }
            ceq - self.sediment
        };

        // Move sediment
        self.sediment += dt * cdiff * deposition_rate;
        let h1 = map.get_mut(self.pos.0, self.pos.1);
        *h1 -= dt * self.volume * deposition_rate * cdiff;

        // Evaporate a bit
        self.volume -= 1.0 - dt * evaporation_rate;
    }

    fn flood(&mut self, map: &mut [Vec<f64>]) {

    }
}

fn hydraulic_erosion_iteration(map: &mut Map<f64>, stream: &mut Map<f64>, pool: &mut Map<f64>, track: &mut Map<bool>, steps: usize) {
    for _ in 0..steps {
        let x = ::rand::random_range(1..=map.dims.0 - 1);
        let y = ::rand::random_range(1..=map.dims.1 - 1);
        let mut drop = WaterUnit::new(x, y);
        drop.descend(map, track);
    }
}

fn do_erosion(map: &mut Map<f64>, stream: &mut Map<f64>, pool: &mut Map<f64>, track: &mut Map<bool>) {
    if true {
        // Very WIP, not great.
        let drop_steps = 1000;
        hydraulic_erosion_iteration(map, stream, pool, track, drop_steps);
    }
}

#[macroquad::main("RPCG")]
async fn main() {
    let mut seed: u32 = ::rand::random();
    let mut w = screen_width();
    let mut h = screen_height();
    //request_new_screen_size(1920., 1080.);
    let mut heightmap = generate_heightmap();

    // Track tells where water ran this iteration
    let mut track = Map::<bool>::zeroed(w as usize, h as usize);

    // Pool tells where pools are in the grid, and how deep they are.
    let mut pool = Map::<f64>::zeroed(w as usize, h as usize);

    // Stream tells where water is moving in the grid, and how deep they are.
    let mut stream = Map::<f64>::zeroed(w as usize, h as usize);

    let before = heightmap.clone();

    // do erosion
    do_erosion(&mut heightmap, &mut stream, &mut pool, &mut track);

    // Textures
    let mut before_texture = heightmap_to_texture(&before);
    let mut texture = heightmap_to_texture(&heightmap);

    let mut screen = 0;

    loop {
        if screen_width() != w || screen_height() != h {
            w = screen_width();
            h = screen_height();
            heightmap = generate_heightmap();
            let before = heightmap.clone();
            do_erosion(&mut heightmap, &mut stream, &mut pool, &mut track);
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

        if is_key_pressed(KeyCode::S) {
            let filename = format!("Island_{seed}.png");
            println!("Saving to {filename}");
            texture.get_texture_data().export_png(&filename);
        }

        if is_key_pressed(KeyCode::Enter) {
            println!("Generating new map...");
            heightmap = generate_heightmap();
            let before = heightmap.clone();
            do_erosion(&mut heightmap, &mut stream, &mut pool, &mut track);
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
